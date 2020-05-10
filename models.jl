using KnetLayers, Statistics, Plots
import Knet: sumabs2, norm, At_mul_B

struct Attention
    key_transform::Dense
    query_transform::Dense
    value_transform::Dense
    attdim::Int
end

function Attention(;memory::Int,query::Int,att::Int)
    Attention(Dense(input=memory,output=att, winit=att_weight_init, activation=Tanh()),
              Dense(input=query, output=att, winit=att_weight_init, activation=Tanh()),
              Dense(input=memory,output=att, winit=att_weight_init,  activation=Tanh()),
              att)
end

function single_attend(m::Attention, memory, query; pdrop=0.0)
    tkey     = m.key_transform(memory)
    tquery   = m.query_transform(query)
    score    = dropout(tquery .* tkey,pdrop) ./ sqrt(m.attdim) #[H,B]
    weights  = softmax(score;dims=1) #[H,B]
    values   = m.value_transform(memory) .* weights # [H,B]
    return (values, weights)
end

function (m::Attention)(memory, query; pdrop=0, mask=nothing, batchdim=3)
    if ndims(memory)  == 2 && ndims(query) == 2
        return single_attend(m, memory,query; pdrop=0)
    end
    tquery  = m.query_transform(query) # [A,N,B]
    if  ndims(tquery) == 2
        tquery = reshape(tquery,m.attdim,1,size(tquery,2))
    end
    if batchdim == 2
        B       = size(memory,batchdim)
        memory  = permutedims(memory, (1,3,2))
        if !isnothing(mask)
            if size(mask,1) != size(memory,2)
                 mask = mask'
            end
        end
    end
    tkey = m.key_transform(memory) #[A,T',B]
    score = dropout(bmm(tquery,tkey,transA=true),pdrop) ./ sqrt(m.attdim) #[N,T',B]
    if !isnothing(mask)
        score = applymask(score, mask, +)
    end
    weights  = softmax(score;dims=2) #[N,T',B]
    values   = bmm(m.value_transform(memory),weights;transB=true) # [M,N,B]
    return mat(values), mat(score), mat(weights)
end

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

struct RNNLM{Data} <: AbstractVAE{Data}
    decoder::LSTM
    output::Linear
    vocab::Vocabulary{Data}
    config::Dict
end

function RNNLM(vocab::Vocabulary{D}, config::Dict; embeddings=nothing) where D <: DataSet
    RNNLM{D}(
        LSTM(input=length(vocab),hidden=config["H"],embed=load_embed(vocab, config, embeddings),numLayers=config["Nlayers"]),
        Linear(input=config["H"],output=length(vocab)),
        vocab,
        config
    )
end

struct ProtoVAE{Data}
    embed::Embed
    decoder::LSTM
    output::Linear
    enclinear::Multiply
    encoder::LSTM
    agendaemb::Linear
    h0
    c0
    source_attention::Attention
    insert_attention::Attention
    delete_attention::Attention
    pw::Pw
    vocab::Vocabulary{Data}
    config::Dict
end


function ProtoVAE(vocab::Vocabulary{T}, config; embeddings=nothing) where T<:DataSet
    att_cnt = config["insert_delete_att"] ? 3 : 1
    config["att_cnt"] = att_cnt
    dinput  = config["E"] + att_cnt*config["attdim"] + config["A"]
    rnndrop = config["pdrop"]
    ProtoVAE{T}(load_embed(vocab, config, embeddings),
                LSTM(input=dinput, hidden=config["H"], dropout=rnndrop, numLayers=config["Nlayers"]),
                Linear(input=att_cnt*config["attdim"] + config["H"], output=config["E"]),
                Multiply(input=config["E"], output=config["Z"]),
                LSTM(input=config["E"], hidden=config["H"], dropout=rnndrop, bidirectional=true, numLayers=config["Nlayers"]),
                Linear(input=2config["H"]+2config["Z"], output=config["A"]),
                Param(zeroarray(arrtype,config["H"],config["Nlayers"])),
                Param(zeroarray(arrtype,config["H"],config["Nlayers"])),
                Attention(memory=2config["H"],query=config["H"], att=config["attdim"]),
                Attention(memory=config["E"],query=config["H"], att=config["attdim"]),
                Attention(memory=config["E"],query=config["H"], att=config["attdim"]),
                Pw{Float32}(2config["Z"], config["Kappa"]),
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

function train!(model::RNNLM, data; dev=nothing, eval=false, returnlist=false, o...)
    ppl = typemax(Float64)
    if !eval
        bestparams = deepcopy(parameters(model))
        setoptim!(model,model.config["optim"])
        model.config["rpatiance"] = model.config["patiance"]
        model.config["gradnorm"]  = 0.0
    else
        losses = []
    end
    total_iter = 0
    lss, ntokens, ninstances = 0.0, 0.0, 0.0
    for i=1:(eval ? 1 : model.config["epoch"])
        lss, ntokens, ninstances = 0.0, 0.0, 0.0
        dt  = Iterators.Stateful((eval ? data : shuffle(data)))
        msg(p) = string(@sprintf("Iter: %d,Lss(ptok): %.2f,Lss(pinst): %.2f, PPL(test): %.2f", total_iter, lss/ntokens, lss/ninstances, ppl))
        for i in progress(msg, 1:((length(dt)-1) ÷ model.config["B"])+1)
            total_iter += 1
            d = getbatch(model, dt, model.config["B"])
            if !eval
                J = @diff loss(model, d)
                if total_iter < 50
                    model.config["gradnorm"] = max(model.config["gradnorm"], 2*clip_gradient_norm_(J, Inf))
                else
                    clip_gradient_norm_(J, model.config["gradnorm"])
                end
                for w in parameters(J)
                    g = grad(J,w)
                    if !isnothing(g)
                        KnetLayers.update!(value(w), g, w.opt)
                    end
                end
            else
                if returnlist
                    ls = loss(model, d; eval=true, returnlist=true)
                    append!(losses,ls[d.unsort])
                    J = mean(ls)
                else
                    J = loss(model, d; eval=true, returnlist=false)
                end
            end
            b           = first(d.x.batchSizes)
            n           = sum(d.xpadded.mask[:,2:end])#length(d.x.tokens) + b
            lss        += value(J)*b
            ntokens    += n
            ninstances += b
        end

        if !isnothing(dev)
            newppl = calc_ppl(model, dev)[1]
            @show newppl
            if newppl > ppl
                lrdecay!(model, model.config["lrdecay"])
                model.config["rpatiance"] = model.config["rpatiance"] - 1
                println("patiance decay, rpatiance: $(model.config["rpatiance"])")
                model.config["rpatiance"] == 0 && break
            else
                for (best,current) in zip(bestparams,parameters(model))
                    copyto!(value(best),value(current))
                end
                ppl = newppl
            end
        else
            println((loss=lss/ntokens,))
        end
        if eval
            ppl = exp(lss/ntokens)
            @show ppl
            if returnlist
                return losses, data
            end
        end
        total_iter > 400000 && break
    end
    if !isnothing(dev) && !eval
        for (best, current) in zip(bestparams,parameters(model))
            copyto!(value(current),value(best))
        end
    end
    return (ppl=ppl, ptokloss=lss/ntokens, pinstloss=lss/ninstances)
end

function loss(morph::RNNLM, x; average=false, eval=false, returnlist=false)
    bow, eow = specialIndicies.bow, specialIndicies.eow
    #xpadded  = pad_packed_sequence(x, bow, toend=false)
    #ygold    = pad_packed_sequence(x, eow)
    B  = size(x.xpadded.tokens,1)
    y        = decode(morph, x.xpadded.tokens[:,1:end-1])
    if !returnlist
        loss = nllmask(y, x.xpadded.tokens[:,2:end] .* x.xpadded.mask[:,2:end]; average=false) / B #first(x.batchSizes)
    else
        logpy = logp(y; dims=1)
        xinds =  x.xpadded.tokens[:,2:end] .* x.xpadded.mask[:,2:end]
        loss = []
        for i=1:B
            inds = fill!(similar(xinds),0)
            inds[i,:] .= xinds[i,:]
            linds = KnetLayers.findindices(logpy, inds)
            push!(loss,-sum(logpy[linds]))
        end
    end
    return loss
end

function decode(morph::RNNLM, x=nothing; sampler=sample)
    H, T = hiddensize(morph), eltype(morph)
    if isnothing(x)
         B = morph.config["B"]
         h = zeroarray(arrtype, H, B, numlayers(morph))
         c = fill!(similar(h),0)
         input  = specialIndicies.bow*ones(Int,B)
         preds  = zeros(Int, B, morph.config["maxLength"])
         for i=1:morph.config["maxLength"]
            out   = morph.decoder(input, h, c; batchSizes=[B], hy=true, cy=true)
            h,c   = out.hidden, out.memory
            input = vec(mapslices(sampler, convert(Array,morph.output(out.y)), dims=1)) #FIXME: Array
            preds[:,i] = input
         end
        return preds
    else
        B  = size(x,1)
        #h  = zeroarray(arrtype,H, B, numlayers(morph))
        #c  = fill!(similar(h),0)
        y  = morph.decoder(x).y
        yf = dropout(y, morph.config["pdrop"])
        morph.output(yf)
    end
end

calc_ppl(model::RNNLM, data; o...) = train!(model,data;eval=true, o...)
calc_mi(model::RNNLM, data) = nothing
calc_au(model::RNNLM, data) = nothing
sampleinter(model::RNNLM, data) = nothing

function sample(model::RNNLM, data=nothing; sampler=sample, o...)
    oldb = model.config["B"]
    B = model.config["B"] = max(oldb, 128)
    vocab = model.vocab
    samples = []
    for k = 1:((model.config["N"]-1) ÷ B)+1
        preds    = decode(model; sampler=sampler)
        predstr  = mapslices(x->trim(x,model.vocab), preds, dims=2)
        predenc  = [trimencoded(preds[i,:]) for i=1:B]
        probs2D  = ones(B,1)
        for i=1:B
            push!(samples,(sample=predstr[i],sampleenc=predenc[i],probs=probs2D[i,:]))
        end
    end
    model.config["B"] = oldb
    samples
end

function encodeID(model::ProtoVAE, I, Imask)
    if isempty(I)
        zeroarray(arrtype,latentsize(model)÷2,size(I,1),1)
    else
        applymask(model.embed(I), Imask, *)
    end
end

function encode(m::ProtoVAE, xp, ID; prior=false)
    inserts   = encodeID(m, ID.I, ID.Imask)
    deletes   = encodeID(m, ID.D, ID.Dmask)
    if !m.config["kill_edit"]
        if prior
            μ     = zeroarray(arrtype,2latentsize(m),size(xp.tokens,1))
        else
            insert_embed =  m.enclinear(mat(sum(inserts, dims=3), dims=1))
            delete_embed =  m.enclinear(mat(sum(deletes, dims=3), dims=1))
            μ     = vcat(insert_embed, delete_embed)
        end
        z         = sample_vMF(μ, m.config["eps"], m.config["max_norm"], m.pw; prior=prior)
    else
        z         = zeroarray(arrtype,2latentsize(m),size(inserts,2))
    end
    #xp_tokens = reshape(xp.tokens,length(xp.tokens),1)
    #token_emb = mat(m.embed(xp_tokens),dims=1)
    #pout      = m.encoder(token_emb; hy=true, batchSizes=xp.batchSizes)
    #proto_emb = mat(pout.hidden,dims=1)
    source_context = m.encoder(m.embed(xp.tokens)).y
    source_embed   = source_hidden(source_context,xp.lengths)
    agenda         = m.agendaemb(vcat(source_embed, z))
    #inds           = _batchSizes2indices(xp.batchSizes)
    #pcontext      = PadRNNOutput2(pout.y, inds)
    return agenda, source_context, z, (I=inserts, D=deletes, Imask=ID.Imask, Dmask=ID.Dmask)
end

function source_hidden(source_context,lengths)
    reshape(cat1d((source_context[:,i,lengths[i]] for i=1:length(lengths))...),size(source_context,1),length(lengths))
end

function vmfKL(m::ProtoVAE)
    k, d = m.config["Kappa"], 2m.config["Z"]
    k*((besseli(d/2.0+1.0,k) + besseli(d/2.0,k)*d/(2.0*k))/besseli(d/2.0, k) - d/(2.0*k)) +
    d * log(k)/2.0 - log(besseli(d/2.0,k)) -
    loggamma(d/2+1) - d * log(2)/2
end

function kl_calc(m)
    """evaluate KL penalty terms for a given model configuration."""
    kl_edit_vec = vmfKL(m)
    norm_term   = log(m.config["max_norm"] / m.config["eps"])
    kl_term     = (1.0-m.config["kill_edit"])*(kl_edit_vec + norm_term)
    return kl_term
end

function print_ex_samples(model, data; beam=true)
    println("generating few examples")
    #for sampler in (sample, argmax)
    for prior in (true, false)
        println("Prior: $(prior) , attend_pr: 0.0")
        for s in sample(model, data; N=10, sampler=sample, prior=prior, sanitize=true, beam=beam)
            println("===================")
            for field in propertynames(s)
                println(field," : ", getproperty(s,field))
            end
            println("===================")
        end
    end
    println("done")
    #end
end

function train!(model::ProtoVAE, data; eval=false, dev=nothing, trnlen=1, returnlist=false)
    ppl = typemax(Float64)
    if !eval
        bestparams = deepcopy(parameters(model))
        setoptim!(model,model.config["optim"])
        model.config["rwritedrop"] = model.config["writedrop"]
        model.config["rpatiance"] = model.config["patiance"]
        model.config["gradnorm"]  = 0.0
    else
        data, evalinds = create_ppl_examples(data, 1000) # FIXME: Shuffle?
        @show length(evalinds)
        losses = []
    end
    total_iter, lss, ntokens, ninstances = 0, 0.0, 0.0, 0.0
    for i=1:(eval ? 1 : model.config["epoch"])
        lss, ntokens, ninstances = 0.0, 0.0, 0.0
        dt  = Iterators.Stateful((eval ? data : shuffle(data)))
        msg(p) = string(@sprintf("Iter: %d,Lss(ptok): %.2f,Lss(pinst): %.2f", total_iter, lss/ntokens, lss/ninstances))
        for i in progress(msg, 1:((length(dt)-1) ÷ model.config["B"])+1)
            total_iter += 1
            d = getbatch(model,dt,model.config["B"])
            isnothing(d) && continue
            if !eval
                J = @diff loss(model, d; eval=false)
                if total_iter < 50
                    model.config["gradnorm"] = max(model.config["gradnorm"], 2*clip_gradient_norm_(J, Inf))
                else
                    clip_gradient_norm_(J, model.config["gradnorm"])
                end
                for w in parameters(J)
                    g = grad(J,w)
                    if !isnothing(g)
                        KnetLayers.update!(value(w), g, w.opt)
                    end
                end
            else
                #J = loss(model, d; eval=false)
                ls = loss(model, d; eval=true)
                append!(losses,ls)
                J = mean(ls)
            end
            b           = size(d[1],1)
            n           = sum(d[2][:,2:end])
            lss        += (value(J)*b)
            ntokens    += n
            ninstances += b
            if !eval && i%500==0
                print_ex_samples(model, data)
            end
        end
        if !isnothing(dev)
            newppl = calc_ppl(model, dev; trnlen=trnlen)[1]
            @show newppl
            if newppl > ppl
                lrdecay!(model, model.config["lrdecay"])
                model.config["rpatiance"] = model.config["rpatiance"] - 1
                println("patiance decay, rpatiance: $(model.config["rpatiance"])")
                model.config["rpatiance"] == 0 && break
            else
                for (best,current) in zip(bestparams,parameters(model))
                    copyto!(value(best),value(current))
                end
                ppl = newppl
            end
        else
            println((loss=lss/ntokens,))
        end
        if eval
            s_losses = map(inds->-logsumexp(-losses[inds]), evalinds)
            nwords = sum((sum(data[first(inds)].x .> 4)+1  for inds in evalinds))
            ppl = exp(sum(s_losses .+ kl_calc(model) .+ log(trnlen))/nwords)
            @show ppl
            if returnlist
                return s_losses, kl_calc(model), log(trnlen), data, evalinds
            end
        end
        total_iter > 400000 && break
        # if total_iter % 30000 == 0
        #     GC.gc(); KnetLayers.gc()
        # end
    end
    if !isnothing(dev) && !eval
        for (best, current) in zip(bestparams,parameters(model))
            copyto!(value(current),value(best))
        end
    end
    return (ppl=ppl, ptokloss=lss/ntokens, pinstloss=lss/ninstances)
end

function create_ppl_examples(eset, numex=200)
    test_sentences = unique(map(x->x.x,eset))
    test_sentences = test_sentences[1:min(length(test_sentences),numex)]
    example_inds   = [findall(e->e.x==t,eset) for t in test_sentences]
    data = []; inds = []; crnt = 1
    for ind in example_inds
        append!(data, eset[ind])
        push!(inds, crnt:crnt+length(ind)-1)
        crnt = crnt+length(ind)
    end
    return data,inds
end

calc_ppl(model::ProtoVAE, data; trnlen=nothing) = train!(model, data; eval=true, dev=nothing, trnlen=trnlen)
calc_mi(model::ProtoVAE, data)  = nothing
calc_au(model::ProtoVAE, data)  = nothing
sampleinter(model::ProtoVAE, data) = nothing

function pickprotos(model::ProtoVAE{SIGDataSet}, processed, esets)
    eset    = length(esets) == 3  ? [esets[1];esets[3]] : esets[1]
    set     = map(d->xfield(model.config["task"],d,model.config["conditional"]),eset)
    set     = unique(set)
    inputs  = map(d->join(model.vocab.tokens[d[1:findfirst(s->s==specialIndicies.sep,d)-1]],' '), set)
    outputs = map(d->join(model.vocab.tokens[d[findfirst(s->s==specialIndicies.sep,d)+1:end]],' '), set)
    data = []
    for i=1:length(set)
        push!(data, (x=Int[specialIndicies.bow], xp=set[i], ID=(I=Int[2],D=Int[2])))
    end
    data, Set(inputs), Set(outputs)
end

function pickprotos(model::RNNLM, processed, esets)
    eset    = length(esets) == 3  ? [esets[1];esets[3]] : esets[1]
    set     = map(d->xfield(model.config["task"],d,model.config["conditional"]),eset)
    set     = unique(set)
    inputs  = map(d->join(model.vocab.tokens[d[1:findfirst(s->s==specialIndicies.sep,d)-1]],' '), set)
    outputs = map(d->join(model.vocab.tokens[d[findfirst(s->s==specialIndicies.sep,d)+1:end]],' '), set)
    processed[1], Set(inputs), Set(outputs)
end


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

function preprocess(m::ProtoVAE{T}, train, devs...) where T<:DataSet
    dist = Levenshtein()
    thresh, cond, maxcnt =  m.config["dist_thresh"], m.config["conditional"], m.config["max_cnt_nb"]
    sets = map((train,devs...)) do set
            map(d->xfield(T,d,cond),set)
    end
    words = map(sets) do set
        map(d->String(map(UInt8,d)),set)
    end
    trnwords = first(words)
    adjlists  = []
    for (k,set) in enumerate(words)
        adj = Dict((i,Int[]) for i=1:length(set))
        for i=1:length(set)
            cw         = set[i]
            neighbours = adj[i]
            cnt   = 0
            inds  = randperm(length(trnwords))
            for j in inds
                j == i && continue
                w    = trnwords[j]
                diff = compare(cw,w,dist)
                if diff > thresh
                    push!(neighbours,j)
                    # if k==1
                    #     push!(adj[j],i)
                    # end
                    cnt+=1
                    cnt == maxcnt && break
                end
            end
            if isempty(neighbours) && i != length(set)
                push!(neighbours, rand(inds))
            end
            append!(adj[i],neighbours)
        end
        push!(adjlists, adj)
    end
    map(zip(sets,adjlists)) do (set, adj)
        procset = []
        for (i,d) in enumerate(set)
            protos = first(sets)[adj[i]]
            for p in protos
                push!(procset, (x=d, xp=p, ID=inserts_deletes(d,p)))
            end
        end
        procset
    end
end

function preprocess(m::ProtoVAE{YelpDataSet}, train, devs...)
    return (train,devs...)
end

function beam_decode(model::ProtoVAE, x, proto, ID, agenda; forviz=false)
    T,B           = eltype(arrtype), size(agenda,2)
    H,V,E, attdim = model.config["H"], length(model.vocab.tokens), model.config["E"], model.config["attdim"]
    input         = ones(Int,B,1) .* specialIndicies.bow #BOW
    if model.config["insert_delete_att"]
        contexts   = (batch_last(proto.context), batch_last(ID.I), batch_last(ID.D))
        masks      = (arrtype(proto.mask'*T(-1e18)),  arrtype(.!ID.Imask'*T(-1e18)), arrtype(.!ID.Dmask'*T(-1e18)))
        attentions = ntuple(k->zeroarray(arrtype,attdim,B),3) # zeroarray(arrtype,E,B), zeroarray(arrtype,E,B))
    else
        contexts   = (batch_last(proto.context),)
        masks      = (arrtype(proto.mask'*T(-1e18)),)
        attentions = (zeroarray(arrtype,attdim,B),) # zeroarray(arrtype,E,B), zeroarray(arrtype,E,B))
    end
    #attentions   = (zeroarray(arrtype,H,B), zeroarray(arrtype,E,B), zeroarray(arrtype,E,B))
    states        = (_repeat(model.h0,B,dim=2), _repeat(model.c0,B,dim=2))
    limit         = model.config["maxLength"]
    traces        = [(zeros(1, B), attentions, states, ones(Int, B, limit), input, nothing, nothing)]
    for i=1:limit
        traces = beam_search(model, traces, contexts, proto, masks, agenda; step=i, forviz=forviz)
    end
    outputs     = map(t->t[4], traces)
    probs       = map(t->vec(t[1]), traces)
    score_arr, output_arr = traces[1][6], traces[1][7]
    return outputs, probs, score_arr, output_arr
end

function beam_search(model::ProtoVAE, traces, contexts, proto, masks, z; step=1, forviz=false)
    result_traces = []
    bw = model.config["beam_width"]
    @inbounds for i=1:length(traces)
        probs, attentions, states, preds, cinput, scores_arr, output_arr = traces[i]
        y, attentions, states, scores, weights = decode_onestep(model, attentions, states, contexts, masks, z, cinput)
        step == 1 ? negativemask!(y,1:6) : negativemask!(y,1:2,4)
        yp  = cat_copy_scores(model, y, scores[1])
        out_soft  = convert(Array, softmax(yp,dims=1))
        if model.config["copy"]
            output = log.(sumprobs(copy(out_soft), proto.tokens))
        else
            output = log.(out_soft)
        end
        srtinds = mapslices(x->sortperm(x; rev=true), output,dims=1)[1:bw,:]
        cprobs  = sort(output; rev=true, dims=1)[1:bw,:]
        inputs  = srtinds
        if step > 1
            stopped = findall(p->p==specialIndicies.eow, preds[:,step-1])
            if step > 2
                cprobs[1,stopped] .= 0.0
            end
            cprobs[2:bw,stopped] .= -100
            inputs[:,stopped] .= specialIndicies.eow
        end
        if forviz
            scores_softmaxed = map(Array, softmax(weights))
        else
            scores_softmaxed = nothing
        end
        push!(result_traces, ((cprobs .+ probs), attentions, states, preds, inputs, scores_softmaxed, out_soft, scores_arr, output_arr))
    end

    global_probs     = vcat(map(first, result_traces)...)
    global_srt_inds  = mapslices(x->sortperm(x; rev=true), global_probs, dims=1)[1:bw,:]
    global_srt_probs = sort(global_probs; rev=true, dims=1)[1:bw,:]
    new_traces = []
    for i=1:bw
        probs      = global_srt_probs[i:i,:]
        inds       = map(s->divrem(s,bw),global_srt_inds[i,:] .- 1)
        attentions = [hcat((result_traces[trace+1][2][k][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...) for k in model.config["att_cnt"]]
        states     = [cat((result_traces[trace+1][3][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds))..., dims=2) for k=1:2]
        inputs     = vcat((result_traces[trace+1][5][loc+1,bi] for (bi, (trace, loc)) in enumerate(inds))...)
        if step == 1
            old_preds  = copy(result_traces[1][4])
        else
            old_preds  = vcat((result_traces[trace+1][4][bi:bi,:] for (bi, (trace, _)) in enumerate(inds))...)
        end
        old_preds[:,step] .= inputs

        if forviz
            scores  = ntuple(k->hcat((result_traces[trace+1][6][k][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...),model.config["att_cnt"])
            outsoft = hcat((result_traces[trace+1][7][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...)
            outsoft = reshape(outsoft,size(outsoft)...,1)
            scores  = map(s->reshape(s,size(s)...,1),scores)

            if step == 1
                scores_arr    = scores
                output_arr    = outsoft
            else
                old_scores    = ntuple(k->cat([result_traces[trace+1][8][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2),model.config["att_cnt"])
                old_outputs   = cat([result_traces[trace+1][9][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2)
                scores_arr    = ntuple(i->cat(old_scores[i],scores[i],dims=3),model.config["att_cnt"])
                output_arr    = cat(old_outputs,outsoft,dims=3)
            end
        else
            scores_arr, output_arr = nothing, nothing
        end
        push!(new_traces, (probs,attentions,states,old_preds,old_preds[:,step:step], scores_arr, output_arr))
    end
    return new_traces
end


function decode(model::ProtoVAE, x, proto, ID, agenda; sampler=sample, training=true)
    T,B        = eltype(arrtype), size(agenda,2)
    H,V,E,attdim      = model.config["H"], length(model.vocab.tokens), model.config["E"], model.config["attdim"]
    #h,c       = expand_hidden(model.h0,B), expand_hidden(model.c0,B)
    input      = ones(Int,B,1) .* specialIndicies.bow #BOW
    if model.config["insert_delete_att"]
        contexts   = (batch_last(proto.context), batch_last(ID.I), batch_last(ID.D))
        masks      = (arrtype(proto.mask'*T(-1e18)),  arrtype(.!ID.Imask'*T(-1e18)), arrtype(.!ID.Dmask'*T(-1e18)))
        attentions = ntuple(k->zeroarray(arrtype,attdim,B),3) # zeroarray(arrtype,E,B), zeroarray(arrtype,E,B))
    else
        contexts   = (batch_last(proto.context),)
        masks      = (arrtype(proto.mask'*T(-1e18)),)
        attentions = (zeroarray(arrtype,attdim,B),) # zeroarray(arrtype,E,B), zeroarray(arrtype,E,B))
    end
    states     = (expand_hidden(model.h0,B), expand_hidden(model.c0,B))
    limit      = (training ? size(x,2)-1 : model.config["maxLength"])
    preds      = ones(Int, B, limit)
    outputs    = []
    L = model.config["copy"] ? V + size(proto.tokens,2) : V
    for i=1:limit
         output, attentions, states, scores, weights = decode_onestep(model, attentions, states, contexts, masks, agenda, input)
         if !training
             i == 1 ? negativemask!(output,1:6) : negativemask!(output,1:2,4)
         end
         output  = cat_copy_scores(model, output, scores[1])
         push!(outputs, output)
         if !training
            output        = convert(Array, softmax(output,dims=1))
            if model.config["copy"]
                 output        = sumprobs(output, proto.tokens)
             end
             preds[:,i]    = vec(mapslices(sampler, convert(Array, value(output)),dims=1))
             input         = preds[:,i:i]
         else
             input         = x[:,i+1:i+1]
         end
    end

    return reshape(cat1d(outputs...),L,B,limit), preds
end

function decode_onestep(model::ProtoVAE, attentions, states, contexts, masks, z, input)
    attdrop = model.config["attdrop"]
    e                         = mat(model.embed(input),dims=1)
    xi                        = vcat(e, attentions..., z)
    out                       = model.decoder(xi, states[1], states[2]; hy=true, cy=true)
    h, c                      = out.hidden, out.memory
    hbottom                   = h[:,:,1]
    source_attn, source_score, source_weight = model.source_attention(contexts[1],hbottom; mask=masks[1], pdrop=attdrop)
    if model.config["insert_delete_att"]
        insert_attn, insert_score, insert_weight = model.insert_attention(contexts[2],hbottom;mask=masks[2], pdrop=attdrop)
        delete_attn, delete_score, delete_weight = model.delete_attention(contexts[3],hbottom;mask=masks[3], pdrop=attdrop)
        y  = model.output(vcat(out.y,source_attn,insert_attn,delete_attn))
        yv = At_mul_B(model.embed.weight,y)
        yv, (source_attn, insert_attn, delete_attn), (h,c), (source_score, insert_score, delete_score), (source_weight, insert_weight, delete_weight)
    else
        y  = model.output(vcat(out.y,source_attn))
        yv = At_mul_B(model.embed.weight,y)
        yv, (source_attn,), (h,c), (source_score,), (source_weight,)
    end
end

function id_attention_drop(ID,attend_pr,B)
    I,D,Imask,Dmask = ID
    if attend_pr == 0
        I     = fill!(similar(I,size(I,1),B,1),0)
        D     = fill!(similar(I),0)
        Imask = falses(B,1)
        Dmask = falses(B,1)
    else
        I = dropout(I, 1-attend_pr)
        D = dropout(D, 1-attend_pr)
        Imask = ID.Imask
        Dmask = ID.Dmask
    end
    return (I=I,D=D,Imask=Imask,Dmask=Dmask)
end

function loss(model::ProtoVAE, data; eval=false)
    xmasked, x_mask, xp_masked, xp_mask, ID, copyinds, unbatched = data
    B = length(first(unbatched))
    xp = (tokens=xp_masked,lengths=length.(unbatched[2]))
    agenda, pcontext, z, ID_enc = encode(model, xp, ID; prior=false)
    ID   = id_attention_drop(ID_enc, model.config["attend_pr"], B)
    output, _ = decode(model, xmasked, (tokens=xp_masked, mask=xp_mask, context=pcontext),  ID, agenda)
    if model.config["copy"]
        ytokens, ymask = xmasked[:,2:end], x_mask[:, 2:end]
        if !eval
            ymask = ymask .* (rand(size(ymask)...) .> model.config["rwritedrop"])
        end
        write_indices = findindices(output, (ytokens .* ymask), dims=1)
        probs = softmax(mat(output, dims=1), dims=1) # H X BT

        inds = ind2BT(write_indices, copyinds, size(probs)...)

        if !eval
            marginals = log.(sum(probs[inds.tokens] .* arrtype(inds.mask),dims=2) .+ 1e-12)
            loss = -sum(marginals[inds.sumind]) / B
        else
            unpadded = reshape(inds.unpadded, B, :)
            T = size(unpadded,2)
            loss = []
            for i=1:B
                push!(loss, -sum((log.(sum(probs[unpadded[i,t]])) for t=1:T if !isempty(unpadded[i,t]))))
            end
        end
    else
        if !eval
            loss = nllmask(output,(xmasked[:, 2:end] .* x_mask[:, 2:end]); average=false) ./ B
        else
            logpy = logp(output;dims=1)
            xinds = xmasked[:, 2:end] .* x_mask[:, 2:end]
            loss = []
            for i=1:B
                inds = fill!(similar(xinds),0)
                inds[i,:] .= xinds[i,:]
                linds = KnetLayers.findindices(logpy, inds)
                push!(loss,-sum(logpy[linds]))
            end
            loss
        end
    end
    return loss
end

function sample(model::ProtoVAE, data; N=nothing, sampler=sample, prior=true, sanitize=true, beam=true, forviz=false)
    N  = isnothing(N) ? model.config["N"] : N
    B  = min(model.config["B"],32)
    dt = data
    vocab = model.vocab
    samples = []
    for i = 1 : (N ÷ B) + 1
        b =  min(N,B)
        if (d = getbatch(model,dt,b)) !== nothing
            xmasked, x_mask, xp_masked, xp_mask, ID, copymask, unbatched = d
            b = size(xmasked, 1)
            xp = (tokens=xp_masked,lengths=length.(unbatched[2]))
            agenda, pcontext, z, ID  = encode(model, xp, ID; prior=prior)
            if sanitize
                ID = id_attention_drop(ID,0.0,b)
            end
            if beam
                preds, probs, scores, outputs  = beam_decode(model, xmasked, (tokens=xp_masked, mask=xp_mask, context=pcontext), ID, agenda; forviz=forviz)
                if forviz
                    for i=1:b
                        @inbounds push!(samples, process_for_viz(vocab,
                        preds[1][i,:],
                        xp_masked.tokens[i,:],
                        ID.I[i,:],
                        ID.D[i,:],
                        ntuple(k->scores[k][:,i,:],2),
                        outputs[:,i,:]))
                    end
                    length(samples) >= N && break
                    continue
                end
                predstr = [join(ntuple(k->trim(preds[k][i,:], vocab),length(preds)),'\n')  for i=1:b]
                predenc = [trimencoded(preds[1][i,:]) for i=1:b]
                probs2D = softmax(hcat(probs...), dims=2)
                #s       = hcat(map(pred->vec(mapslices(x->trim(x,vocab), pred, dims=2)), preds)...)
                #s       = mapslices(s->join(s,'\n'),s,dims=2)

            else
                y, preds   = decode(model, xmasked, (tokens=xp_masked, mask=xp_mask, context=pcontext), ID, agenda; sampler=sampler, training=false)
                predstr    = mapslices(x->trim(x,vocab), preds, dims=2)
                probs2D    = ones(b,1)
                predenc    = [trimencoded(preds[i,:]) for i=1:b]
            end
            for i=1:b
                push!(samples, (target  = join(vocab.tokens[unbatched[1][i]],' '),
                                proto    = join(vocab.tokens[unbatched[2][i]],' '),
                                inserts  = vocab.tokens[unbatched[3][i]],
                                deletes  = vocab.tokens[unbatched[4][i]],
                                sample   = predstr[i],
                                sampleenc = predenc[i],
                                probs=probs2D[i,:]))
            end
        end
    end
    return samples
end


function getbatch(model::Union{RNNLM, AbstractVAE}, iter, B)
    edata = collect(Iterators.take(iter,B))
    if (b = length(edata)) != 0
        unk, mask, eow, bow,_,_ = specialIndicies
        sfs, exs, perms = first.(edata), second.(edata), last.(edata)
        sfs = limit_seq_length_eos_bos(sfs; maxL=model.config["maxLength"])
        r   = sortperm(sfs, by=length, rev=true)
        #r   = 1:length(sfs)
        sfs = sfs[r]
        xpadded = PadSequenceArray(sfs, pad=mask, makefalse=false)
        xi = _pack_sequence(sfs)
        xt = map(_pack_sequence,[exs[r]]) #FIXME: I changed exs part!!
        return (x=xi, examplers=xt, perm=perms[r], xpadded=xpadded, unsort=sortperm(r))
    end
    nothing
end

limit_seq_length_eos_bos(x; maxL=30) =
    map(s->(length(s)>maxL ? [specialIndicies.bow; s[1:Int(maxL)]] : [specialIndicies.bow;s;specialIndicies.eow]) , x)

function getbatch(model::ProtoVAE, iter, B)
    edata = collect(Iterators.take(iter,B))
    b = length(edata); b==0 && return nothing
    x, xp, ID = unzip(edata)
    I, D = unzip(ID)
    unk, mask, eow, bow,_,_ = specialIndicies
    maxL = model.config["maxLength"]
    x = limit_seq_length_eos_bos(x;maxL=maxL)
    #x  = map(s->[bow;s], xlimited)
    xp = map(s->s, limit_seq_length(xp;maxL=maxL))
    #x  = map(s->(length(s)>25 ? s[1:25] : s) , x) # FIXME: maxlength as constant
    #xp = map(s->(length(s)>25 ? s[1:25] : s) , xp)
    #xp_packed   = _pack_sequence(xp)
    pxp         = PadSequenceArray(xp, pad=mask, makefalse=true)
    px          = PadSequenceArray(x, pad=mask, makefalse=false)
    # xpmasked    = PadSequenceArray(xp, pad=mask)
    # x_mask      = get_mask_sequence(length.(x) .+ 2; makefalse=false)
    # xp_mask     = get_mask_sequence(length.(xp); makefalse=true)
    # xmasked     = PadSequenceArray(map(xi->[specialIndicies.bow;xi;specialIndicies.eow], x), pad=mask)
    #Imask       = get_mask_sequence(length.(I); makefalse=false)
    pI          = PadSequenceArray(I, pad=mask, makefalse=false)
    #Dmask       = get_mask_sequence(length.(D); makefalse=false)
    pD          = PadSequenceArray(D, pad=mask, makefalse=false)
    V = length(model.vocab.tokens)
    L = V + size(pxp.tokens,2)
    copymask  = copy_indices(xp, px.tokens,  L, V)
    unbatched   = (x, xp, I ,D)
    return (px..., pxp..., (I=pI[1], D=pD[1], Imask=pI[2], Dmask=pD[2]), copymask, unbatched)
end

function create_copy_mask(xp, xp_mask, xmasked)
    mask = trues(size(xmasked,2)-1, size(xp_mask,2), length(xp)) # T x T' x B
    for i=1:size(xmasked,1)
        for t=2:size(xmasked,2)
            token =  xmasked[i,t]
            if token ∉ specialIndicies
                inds  = findall(t->t==token,xp[i])
                mask[t,inds,i] .= false
            end
        end
    end
    return mask
end


# function pickprotos(edata)
#     x, xp, ID = unzip(edata)
#     if first(xp) isa Vector{Int}
#         I,D       = unzip(ID)
#         #xp = x  # uncomment  if you want to debug with copying. config["kill_edit"] = true
#         #r         = sortperm(xp, by=length, rev=true)
#         x, xp, I, D, 1:length(x)
#         #x[r], xp[r], I[r], D[r], sortperm(r)
#     else
#         inds = map(l->rand(1:l),length.(xp))
#         xp   = map(d->d[1][d[2]],zip(xp,inds))
#         ID   = map(d->d[1][d[2]],zip(ID,inds))
#         I,D  = unzip(ID)
#         r    = sortperm(xp, by=length, rev=true)
#         x[r], xp[r], I[r], D[r], sortperm(r)
#     end
# end
