using KnetLayers, Statistics, Plots
import Knet: sumabs2, norm

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
    e      = morph.encoder(xi.tokens; batchSizes=xi.batchSizes, hy=true).hidden
    H,B,_  = size(e)
    if xt != nothing
        hs = (morph.encoder(ex.tokens; batchSizes=ex.batchSizes, hy=true).hidden  for ex  in xt)
        h  = reshape(cat1d(hs...), H,4morph.num + 1, B)
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

function embed_x_concat_z(embed, x, z; concatz = false)
    if concatz
        vcat(embed(x.tokens),z[:, _batchSizes2ids(x)])
    else
        embed(x.tokens)
    end
end

function decode(morph, z, xi=nothing; bow=4, maxL=20, sampler=sample)
    c = morph.Wdec(z)
    h = tanh.(c)
    concatz = morph.decoder.specs.inputSize == (size(morph.dec_embed.weight,1) + size(z,1))
    if isnothing(xi)
         B      = size(h,2)
         input  = fill!(Vector{Int}(undef,B),bow)
         preds  = zeros(Int,B,maxL)
         for i=1:maxL
            if concatz
                e = vcat(morph.dec_embed(input),z)
            else
                e = morph.dec_embed(input)
            end
            out = morph.decoder(e,h,c; hy=true, cy=true)
            h,c = out.hidden, out.memory
            input  = vec(mapslices(sampler, convert(Array,morph.output(out.y)), dims=1))
            preds[:,i] = input
         end
        return preds
    else
        x = pad_packed_sequence(xi, bow, toend=false)
        e = embed_x_concat_z(morph.dec_embed,x,z; concatz=concatz)
        y = morph.decoder(e, h, c; batchSizes=x.batchSizes).y
        morph.output(drop(y))
    end
end

function decodensample(morph, z, xi; bow=4, maxL=20, sampler=sample)
    nz,nsample,B = size(z)
    z  = reshape(z,nz,nsample * B)
    c = morph.Wdec(z)
    h = tanh.(c)
    x,_ = nsample_packed_sequence(xi, bow, toend=false, nsample=nsample)
    concatz = morph.decoder.specs.inputSize == (size(morph.dec_embed.weight,1) + size(z,1))
    e = embed_x_concat_z(morph.dec_embed,x,z; concatz=concatz)
    y = morph.decoder(e, h, c ; batchSizes=x.batchSizes).y
    morph.output(drop(y))
end

function llsingleseq(lp,yt,inds)
    y = lp[:, inds]
    a = yt.tokens[inds]
    sum(y[findindices(y, a, dims=1)])
end

function ppl_iw(morph, vocab, xi, x; nsample=500)
    μ, logσ², _ =  encode(morph, xi, isencatt(morph) ? x : nothing)
    nz, B =size(μ)
    μ = reshape(μ, nz, 1, B)
    logσ² =  reshape(logσ², nz, 1, B)
    z = μ .+ arrtype(randn(eltype(μ),nz, nsample, B)) .* exp.(0.5f0 .* logσ²)
    log_pz_qzx =  iws(μ, logσ², z)
    y  = decodensample(morph, z, xi; bow=vocab.specialIndices.bow)
    yt,inds = nsample_packed_sequence(xi, vocab.specialIndices.eow, nsample=nsample)
    lp = logp(y,dims=1)
    # Calculate log(p(x|z,𝜃)) for each nsamples and for each instance
    log_pxz = [llsingleseq(lp,yt,inds[(j-1)*nsample + i])  for i=1:nsample, j=1:B] |> arrtype
    -sum(logsumexp(log_pxz .+  log_pz_qzx, dims=1) .- Float32(log(nsample)))
end

# Calculate log(p(z)/q(z|x,ɸ)) for each nsamples and for each instance
function iws(μ, logσ², z)
    nz,nsamples,B = size(z)
    σ² = exp.(logσ²)
    dev = z .- μ
    log_qzx = -0.5f0 * sum(dev.^2 ./ σ², dims=1) .- 0.5f0 * (nz * Float32(log(2π)) .+ sum(logσ², dims=1))
    log_pz = sum(-0.5f0 * Float32(log(2π)).- z.^2 ./ 2,dims=1)
    reshape(log_pz .- log_qzx,nsamples,B)
end


function calc_ppl(model, data, vocab; nsample=500, B=16)
    edata = Iterators.Stateful(data)
    lss, nchars, ninstances = 0.0, 0, 0
    while (d = getbatch(edata,B)) !== nothing
        J = ppl_iw(model, vocab, d[1], d[2]; nsample=nsample)
        nchars += length(d[1].tokens)
        ninstances += d[1].batchSizes[1]
        lss += J
    end
    exp(lss/nchars)
end

function loss(morph, vocab, xi, x; klw=1.0f0, fbr=nothing)
    μ, logσ², _ =  encode(morph, xi, isencatt(morph) ? x : nothing)
    z = μ .+ randn!(similar(μ)) .* exp.(0.5f0 .* logσ²)
    y  = decode(morph, z, xi; bow=vocab.specialIndices.bow)
    KL = 0.5f0 .* (μ.^2 .+ exp.(logσ²) .- 1.0f0 .- logσ²)
    if fbr !== nothing
        s = relu.(KL .- fbr)
        KL = sum((KL .* s) ./ (s .+ Float32(1e-20)))
    else
        KL = sum(KL)
    end
    yt = pad_packed_sequence(xi, vocab.specialIndices.eow)
    B  = size(μ,2)
    L  = nllmask(y,yt.tokens; average=false) / B + klw * KL / B
end

function attentions(model, data, vocab;  B=32)
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

function train_ae!(model, train, vocab; dev=nothing, epoch=30, optim=Adam(), B=16)
    setoptim!(model,optim)
    for i=1:epoch
        lss,cnt = 0.0, 0
        edata = Iterators.Stateful(encode(shuffle(train),vocab))
        while (d = getbatch(edata,B)) !== nothing
            J = @diff loss_ae(model, vocab, d[1], d[2])
            lss += value(J); cnt += 1
            for w in KnetLayers.params(J)
                g = grad(J,w)
                if !isnothing(g)
                    KnetLayers.update!(value(w), g, w.opt)
                end
            end
        end
        if !isnothing(dev)

        else
            println((loss=lss/cnt,))
        end
    end
    return model
end

function loss_ae(morph, vocab, xi, x;)
    z, _ , _ =  encode(morph, xi, isencatt(morph) ? x : nothing)
    y        =  decode(morph, z, xi; bow=vocab.specialIndices.bow)
    yt       =  pad_packed_sequence(xi, vocab.specialIndices.eow)
    B        =  size(z,2)
    L        =  nllmask(y,yt.tokens; average=false) / B
end

function train_vae!(model, train, vocab; dev=nothing, epoch=30, optim=Adam(), B=16, kl_weight=0.0f0, kl_rate = 0.1f0, fb_rate=8.0f0)
    setoptim!(model,optim)
    dim_target_rate = isnothing(fb_rate) ?   nothing :  fb_rate / latentsize(model)
    for i=1:epoch
        lss,cnt = 0.0, 0
        edata = Iterators.Stateful(encode(shuffle(train),vocab))
        while (d = getbatch(edata,B)) !== nothing
            J = @diff loss(model, vocab, d[1], d[2]; klw=kl_weight, fbr = dim_target_rate)
            lss += value(J)
            cnt += 1
            for w in KnetLayers.params(J)
                KnetLayers.update!(value(w), grad(J,w), w.opt)
            end
        end
        kl_weight = Float32(min(1.0, kl_weight+kl_rate))
        if !isnothing(dev)

        else
            println((kl_weight=kl_weight, fbr=fb_rate, loss=lss/cnt))
        end
    end
end

function train_rnnlm!(model, train, vocab; dev=nothing, epoch=30, optim=Adam(), B=16)
    setoptim!(model,optim)
    for i=1:epoch
        lss,cnt = 0.0, 0
        edata = Iterators.Stateful(encode(shuffle(train), vocab))
        while (d = getbatch(edata,B)) !== nothing
            J = @diff losslm(model, vocab, d[1])
            lss += value(J)
            cnt += 1
            for w in KnetLayers.params(J)
                KnetLayers.update!(value(w), grad(J,w), w.opt)
            end
        end
        if !isnothing(dev)

        else
            println((loss=lss/cnt,))
        end
    end
end

function samplingparams(model, data; useprior=false, B=16)
    H,T   = latentsize(model), elementtype(model)
    μ, σ² = fill!(arrtype(undef,H,1),0), fill!(arrtype(undef,H,1),1)
    if useprior
        μ, σ²
    else
        fill!(σ²,0)
        cnt = 0
        edata = Iterators.Stateful(data)
        while ((d = getbatch(edata,B)) !== nothing)
            μi, logσ² = encode(model, d[1], isencatt(model) ? d[2] : nothing)
            μ  .+= sum(μi,dims=2)
            σ² .+= sum(exp.(logσ²),dims=2)
            cnt += size(μi,2)
        end
        μ/cnt, sqrt.(σ²/(cnt-1))
    end
end

function calc_au(model,data; delta=0.01, B=16)
    H, T = latentsize(model), elementtype(model)
    μ    = fill!(arrtype(undef, H,1),0)
    cnt  = 0
    edata = Iterators.Stateful(data)
    while ((d = getbatch(edata,B)) !== nothing)
            μi,_ = encode(model, d[1], isencatt(model) ? d[2] : nothing)
            μ   .+= sum(μi,dims=2)
            cnt  += size(μi,2)
    end
    μavg =  μ/cnt
    cnt=0
    var = fill!(arrtype(undef, H,1),0)
    edata = Iterators.Stateful(data)
    while ((d = getbatch(edata,B)) !== nothing)
        μi, _= encode(model, d[1], isencatt(model) ? d[2] : nothing)
        var  .+= sum((μi .-  μavg).^2, dims=2)
        cnt   += size(μi,2)
    end
    au_var  = convert(Array, var/(cnt-1))
    return sum(au_var .>= delta), au_var,  μavg
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
        neg_entropy += sum(-0.5f0 * nz * Float32(log(2π)) .- 0.5f0 .* sum(1f0 .+ logσ², dims=1))
        push!(mu_batch_list, convert(Array, μ))
        push!(logvar_batch_list, convert(Array,logσ²))
    end
    neg_entropy = neg_entropy / cnt
    log_qz = 0.0
    mu      =  arrtype(reshape(hcat((mu_batch_list[i] for i in  1:length(mu_batch_list) )...),nz,cnt,1))
    logvar  =  arrtype(reshape(hcat((logvar_batch_list[i] for i in 1:length(logvar_batch_list))...),nz,cnt,1))
    var     =  exp.(logvar)
    cnt2    = 0
    for i=1:length(mu_batch_list)
        μ = arrtype(mu_batch_list[i])
        nz,B = size(μ)
        logσ² = arrtype(logvar_batch_list[i])
        z_samples = μ .+ randn!(similar(μ)) .* exp.(0.5f0 .* logσ²)
        cnt2 += B
        z_samples = reshape(z_samples, nz,1,B)
        dev = z_samples .- mu
        log_density = -0.5f0 * sum(dev.^2 ./ var, dims=1) .- 0.5f0 * (nz * Float32(log(2π)) .+ sum(logvar, dims=1))
        log_density = reshape(log_density,cnt,B)
        log_qz += sum(logsumexp(log_density, dims=1) .- log(cnt))
    end
    log_qz /= cnt2
    mi = neg_entropy - log_qz
    return mi
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


function sampleinter(model, vocab, data; N=5, useprior=true)
    edata = Iterators.Stateful(encode(shuffle(data),vocab))
    d = getbatch(edata,2)
    μ, logσ² = encode(model, d[1], isencatt(model) ? d[2] : nothing)
    r     =  similar(μ,size(μ,1),2)
    z     =  μ .+ randn!(r) .* exp.(0.5 .* logσ²)
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
numlayers(model)   = Int(model.decoder.specs.numLayers)
hiddensize(model)  = model.hiddenSize
latentsize(model)  = model.latentSize
elementtype(model) = eltype(value(first(params(model))))
isencatt(model) = haskey(model, :Wμa) &&  haskey(model, :Wσa)
function VAE(V, num; H=512, E=16, Z=16, concatz=false, pdrop=0.4)
    encoder      = LSTM(input=V,hidden=H,embed=E)
    dec_embed    = Embed(input=V,output=E)
    decoder      = LSTM(input=(concatz ? E+Z : E),hidden=H,dropout=pdrop)
    transferto!(dec_embed, encoder.embedding)
    return (encoder=encoder,
            Wμ=Multiply(input=H, output=Z),
            Wσ=Dense(input=H, output=Z, activation=ELU()),
            output=Multiply(input=H,output=V),
            Wdec=Multiply(input=Z, output=H),
            decoder = decoder,
            dec_embed = dec_embed,
            num=num,
            latentSize=Z,
            hiddenSize=H)
end

function EncAttentiveVAE(V, num; H=512, E=16, Z=16, concatz=false, pdrop=0.4)
    encoder    = LSTM(input=V,hidden=H,embed=E)
    dec_embed  = Embed(input=V,output=E)
    decoder    = LSTM(input=(concatz ? E+Z : E),hidden=H,dropout=pdrop)
    transferto!(dec_embed, encoder.embedding)
    return (encoder=encoder,
             Wμ=Multiply(input=H, output=Z), #MLP(H, H ÷ 2, Z, activation=ELU()),
             Wσ=Dense(input=H, output=Z, activation=ELU()), #MLP(H, H ÷ 2, Z, activation=ELU()),
             Weaμ=Linear(input=H, output=H),
             Weaσ=Linear(input=H, output=H),
             Wμa=Linear(input=H, output=H),
             Wσa=Linear(input=H, output=H),
             output=Linear(input=H,output=V),
             Wdec=Multiply(input=Z,output=H),
             decoder = decoder,
             dec_embed = dec_embed,
             num=num,
             latentSize=Z,
             hiddenSize=H)
end

function LSTM_LM(V; H=512, E=16, L=1)
    lstm = LSTM(input=V,hidden=H,embed=E,dropout=0.4,numLayers=L)
    (decoder=lstm, output=Linear(input=H,output=V), hiddenSize=H)
end

function losslm(morph, vocab, xi; average=true)
    y  = decodelm(morph, xi; bow=vocab.specialIndices.bow)
    yt = pad_packed_sequence(xi, vocab.specialIndices.eow)
    nllmask(y,yt.tokens; average=average)
end

# FIXME: Do Multi Layer LM
function decodelm(morph, xi=nothing; bow=4, maxL=20, batch=0, sampler=sample)
    H = hiddensize(morph)
    T = eltype(morph)
    if isnothing(xi)
         B = batch
         h = fill!(arrtype(undef, H, B, numlayers(morph)),0)
         c = fill!(similar(h),0)
         input  = fill!(Vector{Int}(undef,B),bow)
         preds  = zeros(Int, B, maxL)
         for i=1:maxL
            out = morph.decoder(input, h, c; batchSizes=[B], hy=true, cy=true)
            h,c = out.hidden, out.memory
            input  = vec(mapslices(sampler, convert(Array,morph.output(out.y)), dims=1))
            preds[:,i] = input
         end
        return preds
    else
        B = xi.batchSizes[1]
        h = fill!(arrtype(undef, H, B, numlayers(morph)),0)
        c = fill!(similar(h),0)
        x = pad_packed_sequence(xi, bow, toend=false)
        y = morph.decoder(x.tokens, h, c; batchSizes=x.batchSizes).y
        morph.output(drop(y))
    end
end


function calc_ppllm(model, data, vocab; B=16)
    edata = Iterators.Stateful(data)
    lss, nchars, ninstances = 0.0, 0, 0
    while (d = getbatch(edata,B)) !== nothing
        J = losslm(model, vocab, d[1]; average=false)
        nchars += length(d[1].tokens)
        ninstances += d[1].batchSizes[1]
        lss += J
    end
    exp(lss/nchars)
end


function samplelm(model, vocab; N=16, B=16)
    samples = []
    for i = 1 : (N ÷ 16) +1
        y     = decodelm(model; batch=B)
        s     = mapslices(x->trim(x,vocab),y, dims=2)
        push!(samples,s)
    end
    cat1d(samples...)
end

struct Pw{T<:AbstractFloat}
    p::T
    K::T
    a::T
    b::T
    d::T
    beta
end

function Pw{T}(p,K) where T
    dim = p-1
    a = (dim + 2K + sqrt(4K^2 + dim^2))/4
    b = (-2K  + sqrt(4K^2 + dim^2))/dim
    d = 4a*b/(1+b) - dim*log(dim)
    beta = Beta(dim/2, dim/2)
    return Pw(T(p),T(K),T(a),T(b),T(d),beta)
end

function sample(pw::Pw{T}) where T
    opb = (1+pw.b)
    omb = (1-pw.b)
    while true
        β  = rand(pw.beta)
        xp = (1-opb*β)
        xn = (1-omb*β)
        t  = 2*pw.a*pw.b / xn
        if (pw.p-1)*log(t) - t + pw.d - log(rand()) >= 0
            return T(xp / xn)
        end
    end
end



function sample_orthonormal_to(μ, p)
    v      = randn!(similar(μ))
    r      = sum(μ .* v,dims=1)
    ortho  = v .-  μ .* r
    ortho ./ sqrt.(sumabs2(ortho, dims=1))
end

add_norm_noise(μnorm, ϵ, max_norm=7.5f0) =
     min.(μnorm, max_norm-ϵ) .+ rand!(similar(μnorm)) .* ϵ


function sample_vMF(μ, ϵ, max_norm, pw=Pw{eltype(μ)}(size(μ,1),10))
    μ = μ .+ Float32(1e-10)
    w = [sample(pw) for i=1:size(μ,2)]'
    μnorm    = sqrt.(sumabs2(μ,dims=1))
    μnoise   = add_norm_noise(μnorm, ϵ, max_norm)
    v = sample_orthonormal_to(μ ./ μnorm, pw.p)
    scale    = sqrt.(1 .- w.^2)
    μscale = μ .* w ./ μnorm
    (v .* scale  + μscale) .* μnoise
end


function encode(model, x, xp, ID, pw::Pw; max_norm=7.5f0, ϵ=0.1f0)
    μ = vcat(model.enclinear(sum(model.embed(ID.I) .* arrtype(ID.Imask),dims=2)),
             model.enclinear(sum(model.embed(ID.D) .* arrtype(ID.Dmask),dims=2)))
    z = sample_vMF(μ, ϵ, max_norm, pw)
end


function decode(model, x, xp, z; max_norm=7.5f0, ϵ=0.1f0)
    # TODO:
    #bi directional encoder for xp with zero padding: look at macnet
    #encoder with attention # look at macnet
    #look at how to use z in attention or encoder or decoder ?
end
# TODO:
# calculate KL in prototype model
# write loss function for prototype model
# write sampling function for prototype model
# transfer character embeddings from intialization
