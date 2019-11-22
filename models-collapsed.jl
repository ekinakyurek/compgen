using KnetLayers, Statistics
import KnetLayers: nllmask, arrtype
setoptim!(M, optimizer) = for p in params(M); p.opt = deepcopy(optimizer); end
KnetLayers.gc()
gpu(0)

drop(x) = dropout(x, 0.5)
function interact(e, h; sumout=false)
    y = e .* h
    if sumout
        α = softmax(mat(sum(y,dims=1)),dims=1)
        reshape(α,1,size(α)...)
    else
        softmax(y, dims=(1,2))
    end
end

function attend(ea,  h, W, layer; sumout=false) 
    α   = interact(ea,W(h); sumout=sumout)  
    layer(mat(sum(drop(α .* h), dims=2))), α 
end

function encode(morph, xi, xt=nothing) 
    e     = morph.encoder(xi.tokens; batchSizes=xi.batchSizes, hy=true).hidden
    H,B,_ = size(e)
    if xt != nothing
        hs = (morph.encoder(ex.tokens; batchSizes=ex.batchSizes, hy=true).hidden  for ex  in xt)
        h  = reshape(cat1d(hs...), H,4morph.num, B)
        μ, αu = attend(reshape(morph.Weaμ(e),H,1,B), h, morph.Wμa, morph.Wμ; sumout=false) 
        logσ², ασ = attend(reshape(morph.Weaσ(e),H,1,B), h, morph.Wσa,  morph.Wσ; sumout=false)    
        μ, logσ², (αu=αu,ασ=ασ)
    else
        e = reshape(e,H,B)
        μ = morph.Wμ(e)
        logσ² = morph.Wσ(e)
        μ, logσ², nothing
    end
end

function decode(morph, z, xi=nothing; bow=4, maxL=20, sampler=sample)
    c = morph.Wdec(z)
    h = tanh.(c)
    if isnothing(xi) 
         B      = size(h,2)
         H      = hiddensize(morph)
         input  = fill!(Vector{Int}(undef,B),bow)
         preds  = zeros(Int,B,maxL)
         for i=1:maxL
            out      = morph.decoder(input,h,c;batchSizes=[B], hy=true, cy=true)
            h,c = out.hidden, out.memory          
            input  = vec(mapslices(sampler, convert(Array,morph.output(out.y)), dims=1))
            preds[:,i] = input
         end
        return preds
    else
        x = pad_packed_sequence(xi, bow, toend=false)
        y = morph.decoder(x.tokens, reshape(h,size(h)...,1), reshape(c,size(c)...,1) ; batchSizes=x.batchSizes).y
        morph.output(y)
    end
end

function loss(morph, vocab, xi, x; klw=1.0f0, fbr=nothing)
    μ, logσ², _ =  encode(morph, xi, isencatt(morph) ? x : nothing)
    B = size(μ,2)
    z = μ .+ randn!(similar(μ)) .* exp.(0.5f0 .* logσ²)
    y  = decode(morph, z, xi; bow=vocab.specialIndices.bow)
    KL = 0.5f0 * sum(μ.^2 .+ exp.(logσ²) .- 1.0f0 .- logσ²)  
    yt = pad_packed_sequence(xi, vocab.specialIndices.eow)
    L  = (nllmask(y,yt.tokens; average=false) + klw * KL) / B
end

function attentions(model, data, vocab;  B=16)
    edata = Iterators.Stateful(data)
    attentions = []
    while ((d = getbatch(edata,B)) !== nothing)
        xi,xt,perms = d
        μ,logσ²,αs = encode(model, xi, xt)
        sfs = map(inds -> xi.tokens[inds],_batchSizes2indices(xi.batchSizes))
        exs = [map(inds -> x.tokens[inds],_batchSizes2indices(x.batchSizes)) for x in xt]
        push!(attentions, (sfs,exs,perms, map(Array,αs)))
    end
    return attentions
end

function train1!(model, data, vocab; epoch=20, optim=Adam(), B=16, kl_weight=0.0f0, kl_rate = 0.1f0)
    setoptim!(model,optim)
    for i=1:epoch
        lss = 0.0
        cnt = 0
        edata = Iterators.Stateful(encode(shuffle(data),vocab))
        while (d = getbatch(edata,B)) !== nothing
            J = @diff loss(model, vocab, d[1], d[2]; klw=kl_weight)
            lss += value(J)
            cnt += 1
            for w in params(J)
                KnetLayers.update!(value(w), grad(J,w), w.opt)
            end
        end 
        if i>=5
            kl_weight = Float32(min(1.0, kl_weight+kl_rate))
        end
        println((kl_weight=kl_weight, loss=lss/cnt))
    end
end

function samplingparams(model, data; useprior=false, B=16)
    H,T   = latentsize(model), elementtype(model)
    if useprior
        arrtype(zeros(T,H,1)), arrtype(ones(T,H,1))
    else
        μ, σ² = arrtype(zeros(T,H,1)), arrtype(zeros(T,H,1))
        cnt = 0
        edata = Iterators.Stateful(data)
        while ((d = getbatch(edata,B)) !== nothing)
            μi, logσ² = encode(model, d[1], isencatt(model) ? d[2] : nothing)
            μ  .+= sum(μi,dims=2)
            σ² .+= sum(exp.(logσ²),dims=2)
            cnt += size(μi,2)
        end
        μ/cnt, sqrt.(σ²/cnt)
    end
end

function calc_au(model,data; delta=0.01, B=16)
    H,T   = latentsize(model), elementtype(model)
    μ     = arrtype(zeros(T,H,1))
    cnt = 0
    edata = Iterators.Stateful(data)
    while ((d = getbatch(edata,B)) !== nothing)
            μi,_ = encode(model, d[1], isencatt(model) ? d[2] : nothing)
            μ   .+= sum(μi,dims=2)
            cnt  += size(μi,2)
    end
    μavg =  μ/cnt
    cnt=0
    var     = arrtype(zeros(T,H,1))
    edata = Iterators.Stateful(data)
    while ((d = getbatch(edata,B)) !== nothing)
            μi, _= encode(model, d[1], isencatt(model) ? d[2] : nothing)
            var  .+= sum((μi .-  μavg).^2, dims=2)
            cnt   += size(μi,2)
    end
    au_var  = Array(var/(cnt-1))
    return sum(au_var .>= delta), au_var,   μavg
end

function calc_mi(model,data; B=16)
    H,T   = latentsize(model), elementtype(model)
    cnt = 0
    mu_batch_list, logvar_batch_list = [], []
    neg_entropy = 0.0
    edata = Iterators.Stateful(data)
    nz=0
    while ((d = getbatch(edata,B)) !== nothing)
        μ,logσ² = encode(model, d[1], isencatt(model) ? d[2] : nothing)
        nz,B = size(μ)
        cnt  += B
        neg_entropy += sum(-0.5f0 * nz * log(2π) .- 0.5f0 .* sum(1 .+ logσ², dims=2))
        push!(mu_batch_list, convert(Array, μ))
        push!(logvar_batch_list, convert(Array,logσ²))
    end
    neg_entropy = neg_entropy / cnt
    log_qz = 0.0
    mu      =  arrtype(reshape(hcat((mu_batch_list[i] for i in  1:length(mu_batch_list) )...),nz,cnt,1))
    logvar  =  arrtype(reshape(hcat((logvar_batch_list[i] for i in 1:length(mu_batch_list))...),nz,cnt,1))
    var     =  exp.(logvar)
    cnt2 = 0
    for i=1:length(mu_batch_list)
        μ = arrtype(mu_batch_list[i])
        nz,B = size(μ)
        logσ² = arrtype(logvar_batch_list[i])
        z_samples = μ .+ randn!(similar(μ)) .* exp.(0.5f0 .* logσ²)
        cnt2 += B
        z_samples = reshape(z_samples, nz,1,B)
        dev = z_samples .- mu
        log_density = -0.5f0 * sum(dev.^2 ./ var, dims=1) .- 0.5f0 * (nz * log(2π) .+ sum(logvar, dims=1))
        log_density = reshape(log_density,cnt,B)
        log_qz += sum(logsumexp(log_density, dims=1) .- log(cnt))
    end
    log_qz /= cnt
    mi = neg_entropy - log_qz
    return mi
end

function decodensample(morph, z, xi; bow=4, maxL=20, sampler=sample)
    nz,nsample,B = size(z)
    z  = reshape(z,nz,nsample * B)
    c = morph.Wdec(z)
    h = tanh.(c)
    x = nsample_packed_sequence(xi, bow, toend=false, nsample=nsample)
    y = morph.decoder(x.tokens, reshape(h,size(h)...,1), reshape(c,size(c)...,1) ; batchSizes=x.batchSizes).y
    morph.output(drop(y))
end

function ppl(morph, vocab, xi, x; nsample=1000)
    μ, logσ², _ =  encode(morph, xi, isencatt(morph) ? x : nothing)
    nz, B =size(μ)
    μ = reshape(μ, nz, 1, B) 
    logσ² =  reshape( logσ², nz, 1, B) 
    z = μ .+ arrtype(randn(eltype(μ),nz, nsample, B)) .* exp.(0.5f0 .* logσ²)
    y  = decodensample(morph, z, xi; bow=vocab.specialIndices.bow)
    yt,_ = nsample_packed_sequence(xi, vocab.specialIndices.eow, nsample=nsample)
    L  = nllmask(y,yt.tokens; average=false)
    L, length(yt.tokens)
end

    
function calc_ppl(model, data; nsample=1000, B=16)
    lss, cnt = 0.0, 0
    edata = Iterators.Stateful(data)  
    while (d = getbatch(edata,B)) !== nothing
        J, no = ppl(morph, vocab, d[1], d[2]; nsample=nsample) 
        lss += J
        cnt += no
    end
    println((loss=lss/cnt,))
    return model
end

greedy(y) = mapslices(argmax, y, dims=1)
function trim(chars::Vector{Int},vocab)
    out = Int[]
    for c in chars
        c == vocab.specialIndices.eow && break
        if c ∉ vocab.specialIndices
            push!(out,c)
        end
    end
    return join(vocab.chars[out])
    end 

function sample(model, vocab, data; N=5, useprior=true)
    μ, σ =  samplingparams(model, data; useprior=useprior) 
    samples = []
    for i = 1 : (N ÷ 5) +1
        r     =  similar(μ,size(μ,1),5)
        z     =  μ .+ randn!(r) .*  σ
        y     = decode(model, z)
        s     = mapslices(x->trim(x,vocab),y, dims=2)
        push!(samples,s)
    end
    cat1d(samples...)
end


function sampleinter(model, vocab, data; N=5, useprior=false)
    edata = Iterators.Stateful(data)
    d = getbatch(edata,2)
    μ, logσ² = encode(model, d[1], isencatt(model) ? d[2] : nothing)
    #μ, σ² =  samplingparams(model, data; useprior=useprior) 
    r     =  similar(μ,size(μ,1),2)
    z     =  μ .+ randn!(r) .* sqrt.(exp.(logσ²))
    y     = decode(model, z; sampler=argmax)
    ss,se = mapslices(x->trim(x,vocab),y, dims=2)
    zs,ze = z[:,1],z[:,2]
    delta = (ze-zs) ./ N
    samples = [ss]
    for i=1:N
        zi = zs + i*delta
        y  = decode(model, zi; sampler=argmax)
        si = first(mapslices(x->trim(x,vocab),y, dims=2))
        push!(samples,si)
    end
    push!(samples,se)
    return samples
end

function catsample(p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end    
end

sample(y) = catsample(softmax(y;dims=1))

hiddensize(model)  = model.hiddenSize
latentsize(model)  = model.latentSize
elementtype(model) = eltype(model.encoder.params)
isencatt(model) = haskey(model, :Wμa) &&  haskey(model, :Wσa) 

function VAE(V; H=512, E=64, Z=16)
    encoder = LSTM(input=V,hidden=H,embed=E,bidirectional=true, dropout=0.3)
    decoder = LSTM(input=V,hidden=H,embed=E,dropout=0.3)
    decoder.embedding = encoder.embedding
    return (encoder=encoder,
            Wμ=Multiply(input=2H, output=Z), 
            Wσ=Dense(input=2H, output=Z, activation=ELU()), 
            output=Linear(input=H,output=V),
            Wdec=Multiply(input=Z, output=H),
            decoder = decoder, 
            num=edata.num,
            latentSize=Z,
            hiddenSize=H)
end

function EncAttentiveVAE(V; H=512, E=64, Z=16)
    encoder = LSTM(input=V,hidden=H,embed=E,dropout=0.5)
    decoder = LSTM(input=V,hidden=H,embed=E,dropout=0.5)
    decoder.embedding = encoder.embedding
    return (encoder=encoder,
             Wμ=Linear(input=H, output=Z), #MLP(H, H ÷ 2, Z, activation=ELU()), 
             Wσ=Dense(input=H, output=Z, activation=ELU()), #MLP(H, H ÷ 2, Z, activation=ELU()), 
             Weaμ=Linear(input=H, output=H),
             Weaσ=Linear(input=H, output=H),
             Wμa=Linear(input=H, output=H), 
             Wσa=Linear(input=H, output=H),
             output=Linear(input=H,output=V),
             Wdec=Multiply(input=Z,output=H),
             decoder = decoder, 
             num=edata.num,
             latentSize=Z,
             hiddenSize=H)
end
