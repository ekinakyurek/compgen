struct ProtoVAE{Data}
    embed::Embed
    decoder::LSTM
    output::Linear
    enclinear::Multiply
    encoder::LSTM
    agendaemb::Linear
    h0
    c0
    source_attention::PositionalAttention
    insert_attention::PositionalAttention
    delete_attention::PositionalAttention
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
                PositionalAttention(memory=2config["H"],query=config["H"], att=config["attdim"]),
                PositionalAttention(memory=config["E"],query=config["H"], att=config["attdim"]),
                PositionalAttention(memory=config["E"],query=config["H"], att=config["attdim"]),
                Pw{Float32}(2config["Z"], config["Kappa"]),
                vocab,
                config)
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

function train!(model::ProtoVAE, data; eval=false, dev=nothing, trnlen=1, returnlist=false, prior=false)
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
            d = getbatch(model,dt,model.config["B"]; train=!eval)
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
                ls = loss(model, d; eval=true, prior=prior)
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
        contexts   = (proto.context, ID.I,ID.D)
        masks      = (arrtype(proto.mask'*T(-1e18)),  arrtype(.!ID.Imask'*T(-1e18)), arrtype(.!ID.Dmask'*T(-1e18)))
        attentions = ntuple(k->zeroarray(arrtype,attdim,B),3) # zeroarray(arrtype,E,B), zeroarray(arrtype,E,B))
    else
        contexts   = (proto.context,)
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
            scores_softmaxed = map(Array, weights)
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
        contexts   = (proto.context, ID.I, ID.D)
        masks      = (arrtype(proto.mask'*T(-1e18)),  arrtype(.!ID.Imask'*T(-1e18)), arrtype(.!ID.Dmask'*T(-1e18)))
        attentions = ntuple(k->zeroarray(arrtype,attdim,B),3) # zeroarray(arrtype,E,B), zeroarray(arrtype,E,B))
    else
        contexts   = (proto.context,)
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

function loss(model::ProtoVAE, data; eval=false, prior=false)
    xmasked, x_mask, xp_masked, xp_mask, ID, copyinds, unbatched = data
    B = length(first(unbatched))
    xp = (tokens=xp_masked,lengths=length.(unbatched[2]))
    agenda, pcontext, z, ID_enc = encode(model, xp, ID; prior=prior)
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
                #@show [sum(probs[unpadded[i,t]]) |>cpucopy for t=1:T if !isempty(unpadded[i,t])]
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

function viz(model::ProtoVAE, data; N=5, beam=true)
    samples = sample(model, shuffle(data); N=N, sampler=sample, prior=false, beam=beam, forviz=true)
    vocab = model.vocab.tokens
    #json = map(d->JSON.lower((x=vocab[d.x], xp=vocab[d.xp], xpp=vocab[d.xpp], scores1=d.scores[1], scores2=d.scores[2], probs=d.probs)), samples)
    #open("attention_maps.json", "w+") do f

    #    JSON.print(f,json, 4)
    #end
    for i=1:length(samples)
        x, scores, probs, xp = samples[i]
        println("samples: ", join(model.vocab.tokens[x],' '))
        println("xp: ", join(model.vocab.tokens[xp],' '))
        println("----------------------")
        attension_visualize3(model.vocab, probs, scores, xp, x; prefix="$i")
    end
end


function removeunicodes2(toElement)
    vocab = copy(toElement)
    vocab[1:6] = ["<unk>", "<mask>", "<eow>", "<bow>","sep","hash"]
    return vocab
end

function attension_visualize3(vocab, probs, scores, xp, x; prefix="")
    words = removeunicodes2(vocab.tokens.toElement)
    x  = words[x]
    @show y1 = words[xp]
    scale = 1000/15
    attributes = (color=:ice,
                  aspect_ratio=:auto,
                  legend=false,
                  xtickfont=font(9, "Serif"),
                  ytickfont=font(9, "Serif"),
                  xrotation=30,
                  xticks=(1:length(x),x),
                  ticks=:all,
                  size=(length(x)*scale, length(x)*scale),
                  dpi=200,
                )

    # y3 = [words; "xp" .* string.(1:length(xp))]
    # l = @layout [a b]
    p1 = heatmap(scores[1];yticks=(1:length(y1),y1),title="xp-x attention", attributes...)
    # p3 = heatmap(probs;yticks=(1:length(y3),y3), title="action probs", attributes...)
    # p  = Plots.plot(p3,p1; layout=l)
    Plots.savefig(p1, prefix*"_attention_map_proto.pdf")
end

function process_for_viz(vocab, pred, xp, I, D, scores, probs)
    xtrimmed    = trimencoded(pred)
    xps         = (trimencoded(xp),)
    attscores   = [score[1:length(xps[k]),1:length(xtrimmed)] for  (k,score) in enumerate(scores)]
    ixp_end     = length(vocab)+length(xps[1])
    indices     = collect(1:ixp_end)
    outsoft     = probs[indices,1:length(xtrimmed)]
    (x=xtrimmed, scores=attscores, probs=outsoft, xp=xps[1])
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
                        xp_masked[i,:],
                        ID.I[i,:],
                        ID.D[i,:],
                        ntuple(k->scores[k][:,i,:],1),
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

function print_ex_samples(model::ProtoVAE, data; beam=true, mixsampler=false)
    println("generating few examples")
    #for sampler in (sample, argmax)
    for prior in (true, false)
        println("Prior: $(prior) , attend_pr: 0.0")
        for s in sample(model, data; N=10, sampler=argmax, prior=prior, beam=beam)
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

function getbatch(model::ProtoVAE, iter, B; train=true)
    edata = collect(Iterators.take(iter,B))
    b = length(edata); b==0 && return nothing
    x, xp, ID = unzip(edata)
    I, D = unzip(ID)
    if train && model.config["p(xp=x)"] != 0
        for i in findall(rand(length(x)) .< model.config["p(xp=x)"])
            xp[i] = x[i]
            I[i]  = Int[]
            D[i]  = Int[]
        end
    end
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
