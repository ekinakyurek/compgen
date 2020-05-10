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


function pickprotos(model::RNNLM, processed, esets)
    eset    = length(esets) == 3  ? [esets[1];esets[3]] : esets[1]
    set     = map(d->xfield(model.config["task"],d,model.config["conditional"]),eset)
    set     = unique(set)
    inputs  = map(d->join(model.vocab.tokens[d[1:findfirst(s->s==specialIndicies.sep,d)-1]],' '), set)
    outputs = map(d->join(model.vocab.tokens[d[findfirst(s->s==specialIndicies.sep,d)+1:end]],' '), set)
    processed[1], Set(inputs), Set(outputs)
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
