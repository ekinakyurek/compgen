abstract type AbstractVAE{T} end
struct VAE{Data} <: AbstractVAE{Data}
    encoder::LSTM
    Wμ::Multiply
    Wσ::Dense
    output::Multiply
    Wdec::Multiply
    decoder::LSTM
    embed::Embed
    vocab::Vocabulary{Data}
    config
end

function VAE(vocab::Vocabulary{T}, config; embeddings=nothing) where T <: DataSet#V, num; H=512, E=16, Z=16, concatz=false, pdrop=0.4)
    embed    = load_embed(vocab, config, embeddings)
    encoder  = LSTM(input=length(vocab),hidden=config["H"],embed=embed)
    decoder  = LSTM(input=(config["concatz"] ? config["E"]+config["Z"] : config["E"]),hidden=config["H"],dropout=config["pdrop"])
    #transferto!(embed, encoder.embedding)
    VAE{T}(encoder,
        Multiply(input=config["H"],output=config["Z"]),
        Dense(input=config["H"],output=config["Z"],activation=config["activation"]()),
        Multiply(input=config["H"],output=length(vocab)),
        Multiply(input=config["Z"], output=config["H"]),
        decoder,
        embed,
        vocab,
        config)
end

struct EncAttentiveVAE{Data} <: AbstractVAE{Data}
    encoder::LSTM
    Wμ::Multiply
    Wσ::Dense
    Weaμ::Linear
    Weaσ::Linear
    Wμa::Linear
    Wσa::Linear
    output::Multiply
    Wdec::Multiply
    decoder::LSTM
    embed::Embed
    vocab::Vocabulary{Data}
    config::Dict
end

function EncAttentiveVAE(vocab::Vocabulary{T}, config) where T<:DataSet # num; H=512, E=16, Z=16, concatz=false, pdrop=0.4)
    embed      = load_embed(vocab, config, embeddings)
    encoder    = LSTM(input=length(vocab),hidden=config["H"],embed=embed)
    decoder    = LSTM(input=(config["concatz"] ? config["E"]+config["Z"] : config["E"]),hidden=config["H"],dropout=config["pdrop"])
    #transferto!(embed, encoder.embedding)
    EncAttentiveVAE{T}(encoder,
                    Multiply(input=config["H"],output=config["Z"]), #MLP(H, H ÷ 2, Z, activation=ELU()),
                    Dense(input=config["H"],output=config["Z"], activation=config["activation"]()), #MLP(H, H ÷ 2, Z, activation=ELU()),
                    Linear(input=config["H"],output=config["H"]),
                    Linear(input=config["H"],output=config["H"]),
                    Linear(input=config["H"],output=config["H"]),
                    Linear(input=config["H"],output=config["H"]),
                    Multiply(input=config["H"],output=length(vocab)),
                    Multiply(input=config["Z"],output=config["H"]),
                    decoder,
                    embed,
                    vocab,
                    config)
end



function interact(x, h; sumout=true, mask=nothing)
    H,N,B = size(h)
    x     = reshape(x,H,1,B)
    if sumout
        y = applymask(bmm(x,h; transA=true),mask, -) # 1,N,B
        α = reshape(softmax(mat(y),dims=1),1,N,B)
    else
        y = applymask(x .* h, mask, -)
        α = softmax(y, dims=(1,2))
    end
end

function attend(x, projX, h, projH, output, mask=nothing; sumout=false, pdrop=0.4)
    α   = interact(projX(x), projH(h); sumout=sumout, mask=mask)
    v   = dropout(α .* h, pdrop)
    y   = output(mat(sum(v, dims=2)))
    y, α
end

encode_input(m::AbstractVAE, x) =
    m.encoder(x.tokens; batchSizes=x.batchSizes, hy=true).hidden
encode_examplers(m::EncAttentiveVAE, xs, shape) =
    reshape(cat1d((encode_input(m,x) for x in xs)...), shape)

embed_output(m::Embed, x, z::Nothing) = m(x.tokens)
embed_output(m::Embed, x, z) = vcat(m(x.tokens), z[:, _batchSizes2ids(x)])

function encode(m::VAE, x, examplers)
    e   = encode_input(m,x)
    H,B = size(e)
    e   = reshape(e,H,B)
    μ, logσ² = m.Wμ(e), m.Wσ(e)
    μ, logσ², nothing
end

function encode(m::EncAttentiveVAE, x, examplers)
    e         = encode_input(m,x)
    H,B       = size(e)
    h         = encode_examplers(m, examplers, (H,3m.config["num_examplers"],B))
    μ, αu     = attend(e, m.Weaμ, h, m.Wμa, m.Wμ; sumout=false) #FIXME: TRY sumout true again
    logσ², ασ = attend(e, m.Weaσ, h, m.Wσa, m.Wσ; sumout=false)
    μ, logσ², (αu=αu,ασ=ασ)
end

function decode(m::AbstractVAE, z, x::Nothing; sampler=sample)
    c       = m.Wdec(z)
    h       = tanh.(c)
    B       = size(h,2)
    input   = specialIndicies.bow * ones(Int,B)
    preds   = zeros(Int,B,m.config["maxLength"])
    for i=1:m.config["maxLength"]
        e   = m.embed(input)
        xi  = isconcatz(m) ? vcat(e,z) : e
        out = m.decoder(xi, h, c; hy=true, cy=true)
        h,c = out.hidden, out.memory
        input = vec(mapslices(sampler, convert(Array,m.output(out.y)), dims=1)) # FIXME: I don't like the Array here
        preds[:,i] = input
    end
    preds
end

function decode(m::AbstractVAE, z, x;  sampler=sample)
    c  = m.Wdec(z)
    h  = tanh.(c)
    e  = embed_output(m.embed, x, (isconcatz(m) ? z : nothing))
    y  = m.decoder(e, h, c; batchSizes=x.batchSizes).y
    yf = dropout(y, m.config["pdrop"])
    m.output(yf)
end

function ppl_iw(m::AbstractVAE, xi, x=nothing)
    bow, eow    =  specialIndicies.bow, specialIndicies.eow
    nsample     =  m.config["Nsamples"]
    μ, logσ², _ =  encode(m, xi, x)
    nz, B       =  size(μ)
    μ           =  reshape(μ, nz, 1, B)
    logσ²       =  reshape(logσ², nz, 1, B)
    z           =  μ .+ randn!(similar(μ, nz, nsample, B)) .* exp.(0.5 .* logσ²)
    log_imp     =  log_importance(μ, logσ², z)
    xx, _       =  nsample_packed_sequence(xi, bow, toend=false, nsample=nsample)
    ygold, inds =  nsample_packed_sequence(xi, eow, nsample=nsample)
    y           =  decode(m, reshape(z, nz, B*nsample), xx)
    lp          =  logp(y,dims=1)[findindices(y, ygold.tokens, dims=1)]
    log_pxz     =  [sum(lp[inds[(j-1)*nsample+i]]) for i=1:nsample, j=1:B]
    log_pxz     =  convert(arrtype,log_pxz)
    -sum(logsumexp(log_pxz .+  log_imp , dims=1) .- log(nsample))
end

# Calculate log(p(z)/q(z|x,ɸ)) for each nsamples and for each instance
function log_importance(μ, logσ², z)
    nz,nsamples,B = size(z)
    dev = z .- μ
    log_qzx = -0.5 * sum(dev.^2 ./ exp.(logσ²), dims=1) .- 0.5 * (nz * log(2π) .+ sum(logσ², dims=1))
    log_pz = sum(-0.5 * log(2π) .- z.^2 ./ 2,dims=1)
    reshape(log_pz .- log_qzx,nsamples,B)
end

function calc_ppl(m::AbstractVAE, data)
    edata = Iterators.Stateful(data)
    lss, nchars, ninstances = 0.0, 0, 0
    while (d = getbatch(edata,m.config["B"])) !== nothing
        J = ppl_iw(m, d.x, d.examplers)
        nchars += length(d[1].tokens)
        ninstances += d[1].batchSizes[1]
        lss += J
    end
    exp(lss/nchars)
end

function loss(m::AbstractVAE, xi, x; variational=true)
    bow, eow = specialIndicies.bow, specialIndicies.eow
    μ, logσ² = encode(m, xi, x)
    if variational
        z    = μ .+ randn!(similar(μ)) .* exp.(0.5 * logσ²)
    else
        z    = μ
    end
    xpadded  = pad_packed_sequence(xi, bow, toend=false)
    ygold    = pad_packed_sequence(xi, eow)
    y        = decode(m, z, xpadded)
    loss     = nllmask(y, ygold.tokens; average=false)
    if variational
        KL = 0.5 * (μ.^2 .+ exp.(logσ²) .- 1 .- logσ²)
        if m.config["dfbr"] != 0
            KL = freebits(KL,m.config["dfbr"])
        end
        loss += m.config["rklw"] * sum(KL)
    end
    loss / size(μ,2)
end

function attentions(m::EncAttentiveVAE, data)
    edata = Iterators.Stateful(data)
    attentions = []
    while ((d = getbatch(edata,m.config["B"])) !== nothing)
        x,examplers,perms = d
        μ,logσ²,αs = encode(m, x, examplers)
        sfs = map(inds -> x.tokens[inds],_batchSizes2indices(x.batchSizes))
        exs = [map(inds -> ex.tokens[inds],_batchSizes2indices(ex.batchSizes)) for ex in examplers]
        push!(attentions, (sfs,exs,perms, map(Array,αs)))
    end
    return attentions
end

function pretraining_autoencoder(model, data; dev=nothing)
    ae = model.config["model"](model.vocab, model.config)
    train!(ae, data; dev=dev, variational=false)
    for f in (:encoder, :Wμ, :Wσ, :Weaμ, :Weaσ, :Wμa, :Wμσ)
        if isdefined(model,f) && isdefined(ae,f)
            transferto!(getproperty(model,f), getproperty(ae,f))
        end
    end
  # transferto!(model.embed, model.encoder.embedding)
    ae=nothing; GC.gc(); gpugc()
end

function train!(m::AbstractVAE, data; dev=nothing, variational=true) #epoch=30, optim=Adam(), B=16, kl_weight=0.0, kl_rate = 0.1, fb_rate=8.0)
    if variational
        if m.config["aepoch"] != 0
            pretraining_autoencoder(m, data; dev=nothing)
        end
        m.config["rklw"] = m.config["kl_weight"]
        m.config["dfbr"] = m.config["fb_rate"] == 0 ?  nothing :  (m.config["fb_rate"] / latentsize(m))
    end
    epoch = variational ? m.config["epoch"] : m.config["aepoch"]
    bestparams = deepcopy(parameters(m))
    setoptim!(m,m.config["optim"])
    ppl = typemax(Float64)
    m.config["rpatiance"] = m.config["patiance"]
    for i=1:epoch
        lss, nchars, ninstances = 0.0, 0, 0
        edata = Iterators.Stateful(shuffle(data)) #encode(shuffle(data),m.vocab,m.config)) #FIXME: This makes me feel bad
        while (d = getbatch(edata,m.config["B"])) !== nothing
            J           = @diff loss(m, d.x, d.examplers; variational=variational)
            b           = first(d[1].batchSizes)
            n           = length(d[1].tokens) + b
            lss        += value(J)*b
            nchars     += n
            ninstances += b
            for w in parameters(J)
                g = grad(J,w)
                if !isnothing(g)
                    KnetLayers.update!(value(w), g, w.opt)
                end
            end
        end
        if variational
            m.config["rklw"] = min(1, m.config["rklw"]+m.config["kl_rate"])
        end
        if !isnothing(dev)
            newppl = calc_ppl(m, dev)
            @show newppl
            @show lss/nchars
            if newppl > ppl
                lrdecay!(m, m.config["lrdecay"])
                m.config["rpatiance"] = m.config["rpatiance"] - 1
                println("patiance decay, rpatiance: $(m.config["rpatiance"])")
                if m.config["rpatiance"] == 0
                    break
                end
            else
                for (best,current) in zip(bestparams,parameters(m))
                    copyto!(value(best),value(current))
                end
                ppl = newppl
            end
        else
            println((loss=lss/nchars,))
        end
    end
    if !isnothing(dev)
        for (best, current) in zip(bestparams,parameters(m))
            copyto!(value(current),value(best))
        end
    end
end

function samplingparams(model::AbstractVAE, data; useprior=false, B=16)
    H,T   = latentsize(model), elementtype(model)
    μ, σ² = zeroarray(arrtype,H,1), zeroarray(arrtype,H,1;fill=1)
    if useprior
        μ, σ²
    else
        fill!(σ²,0)
        cnt = 0
        edata = Iterators.Stateful(data)
        while ((d = getbatch(edata,model.config["B"])) !== nothing)
            μi, logσ² = encode(model, d.x, d.examplers)
            μ  .+= sum(μi,dims=2)
            σ² .+= sum(exp.(logσ²),dims=2)
            cnt += size(μi,2)
        end
        μ/cnt, sqrt.(σ²/(cnt-1))
    end
end

function calc_au(model::AbstractVAE, data)
    H, T = latentsize(model), elementtype(model)
    μ    = zeroarray(arrtype, H,1)
    cnt  = 0
    edata = Iterators.Stateful(data)
    while ((d = getbatch(edata,model.config["B"])) !== nothing)
            μi,_ = encode(model, d.x, d.examplers)
            μ   .+= sum(μi,dims=2)
            cnt  += size(μi,2)
    end
    μavg =  μ/cnt
    cnt=0
    var = zeroarray(arrtype, H, 1)
    edata = Iterators.Stateful(data)
    while ((d = getbatch(edata,model.config["B"])) !== nothing)
        μi, _= encode(model, d.x, d.examplers)
        var  .+= sum((μi .-  μavg).^2, dims=2)
        cnt   += size(μi,2)
    end
    au_var  = convert(Array, var/(cnt-1))
    return sum(au_var .>= model.config["authresh"]), au_var,  μavg
end

function calc_mi(model::AbstractVAE, data)
    H,T   = latentsize(model), elementtype(model)
    cnt = 0
    mu_batch_list, logvar_batch_list = [], []
    neg_entropy = 0.0
    edata = Iterators.Stateful(data)
    nz=0
    while ((d = getbatch(edata,model.config["B"])) !== nothing)
        μ,logσ² = encode(model, d.x, d.examplers)
        nz,B = size(μ)
        cnt  += B
        neg_entropy += sum(-0.5 * nz * log(2π) .- 0.5 .* sum(1 .+ logσ², dims=1))
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
        z_samples = μ .+ randn!(similar(μ)) .* exp.(0.5 .* logσ²)
        cnt2 += B
        z_samples = reshape(z_samples, nz,1,B)
        dev = z_samples .- mu
        log_density = -0.5 * sum(dev.^2 ./ var, dims=1) .- 0.5 * (nz * log(2π) .+ sum(logvar, dims=1))
        log_density = reshape(log_density,cnt,B)
        log_qz += sum(logsumexp(log_density, dims=1) .- log(cnt))
    end
    log_qz /= cnt2
    mi = neg_entropy - log_qz
    return mi
end

function sample(model::AbstractVAE, data)
    μ, σ =  samplingparams(model, data; useprior=model.config["useprior"])
    samples = []
    for i = 1 : (model.config["N"] ÷ model.config["B"]) +1
        r     =  similar(μ,size(μ,1),model.config["B"])
        z     =  μ .+ randn!(r) .*  σ
        y     =  decode(model, z, nothing)
        s     =  mapslices(x->trim(x,model.vocab),y, dims=2)
        push!(samples,s)
    end
    cat1d(samples...)
end


function sampleinter(model::AbstractVAE, data)
    edata = Iterators.Stateful(shuffle(data))
    d = getbatch(edata,2)
    μ, logσ² = encode(model, d.x, d.examplers)
    r     =  similar(μ,size(μ,1),2)
    z     =  μ .+ randn!(r) .* exp.(0.5 .* logσ²)
    y     = decode(model, z, nothing; sampler=argmax)
    ss,se = mapslices(x->trim(x,model.vocab),y, dims=2)
    zs,ze = z[:,1],z[:,2]
    delta = (ze-zs) ./ model.config["Ninter"]
    samples = [ss]
    for i=1:model.config["Ninter"]
        zi = zs + i*delta
        y  = decode(model, zi, nothing; sampler=argmax)
        si = first(mapslices(x->trim(x,model.vocab),y, dims=2))
        push!(samples,si)
    end
    push!(samples,se)
    return samples
end

numlayers(model)   = model.config["Nlayers"]
hiddensize(model)  = model.config["H"]
latentsize(model)  = model.config["Z"]
embedsize(model)   = model.config["E"]
decinputsize(model)= model.decoder.specs.inputSize
isconcatz(model)   = model.config["concatz"]
elementtype(model) = Float32
isencatt(model)    = haskey(model, :Wμa) &&  haskey(model, :Wσa)


function preprocess(m::AbstractVAE{SIGDataSet}, train, devs...)
    lemma2loc, morph2loc = CrossDict(), CrossDict()
    for (i,datum) in enumerate(train)
          push!(get!(lemma2loc, datum.lemma, Int[]),i)
          push!(get!(morph2loc, datum.tags, Int[]), i)
    end
    L    = length(train)
    num  = m.config["num_examplers"]
    cond = m.config["conditional"]
    sets = map((train, devs...)) do set
        map(set) do d
            s1 =  train[randchoice(get(lemma2loc,d.lemma,Int[]), num , L)]
            s2 =  train[randchoice(get(morph2loc,d.tags,Int[]), num, L)]
            s3 =  train[rand(1:L,num)]
            examplars = map(x->xfield(SIGDataSet,x,cond),[s1;s2;s3])
            x = xfield(SIGDataSet,d,cond)
            r = sortperm(examplars,by=length,rev=true)
            examplars = examplars[r]
            (x=x, examplars=examplars, r=sortperm(r))
        end
    end
end

function preprocess(m::AbstractVAE{SCANDataSet}, train, devs...)
    out2loc = CrossDict()
    for (i,datum) in enumerate(train)
          push!(get!(out2loc, datum.output, Int[]),i)
    end
    L    = length(train)
    num  = m.config["num_examplers"]
    cond = m.config["conditional"]
    sets = map((train, devs...)) do set
        map(set) do d
            s1 =  train[randchoice(get(out2loc,d.output,Int[]), num , L)]
            s2 =  train[rand(1:L,2num)]
            examplars = map(x->xfield(SCANDataSet,x,cond),[s1;s2])
            x = xfield(SCANDataSet,d,cond)
            r = sortperm(examplars,by=length,rev=true)
            examplars = examplars[r]
            (x=x, examplars=examplars, r=sortperm(r))
        end
    end
end
