struct Recombine{Data}
    encembed::Embed
    decembed::Embed
    decoder::LSTM
    combine::Linear
    output::Linear
    copygate::Union{Dense,Nothing}
    copygate2::Union{Dense,Nothing}
    enclinear::Multiply
    xp_encoder::LSTM
    x_encoder::LSTM
    x_xp_inter::PositionalAttention
#   x_xpp_inter::Attention
    z_emb::Linear
    h0
    c0
    attentions
    pw::Pw
    vocab::Vocabulary{Data}
    config::Dict
end

function Recombine(vocab::Vocabulary{T}, config; embeddings=nothing) where T<:DataSet
    dinput = config["E"]  + 2config["Z"] + (get(config,"feedcontext",false) ? config["H"] : 0)
    myinit = linear_init(config["H"])
    lstminit = (winit=myinit, binit=myinit, finit=ones)
    aij_k  = Embed(param(config["attdim"],2*config["Kpos"]+1; init=torchinit, atype=arrtype))
    aij_v  = Embed(param(config["attdim"],2*config["Kpos"]+1; init=torchinit, atype=arrtype))
    p1     = PositionalAttention(memory=2config["H"], query=config["H"], att=config["attdim"], aij_k=aij_k, aij_v=aij_v, normalize=true)
    enc1   = LSTM(;input=config["E"], hidden=config["H"],  bidirectional=true, numLayers=config["Nlayers"], lstminit...)
    if config["seperate"]
        aij_k2 = Embed(param(config["attdim"],2*config["Kpos"]+1;init=torchinit, atype=arrtype))
        aij_v2 = Embed(param(config["attdim"],2*config["Kpos"]+1;init=torchinit, atype=arrtype))
        p2     = PositionalAttention(memory=2config["H"],query=config["H"], att=config["attdim"], aij_k=aij_k2, aij_v=aij_v2, normalize=true)
        enc2   = LSTM(;input=config["E"], hidden=config["H"], bidirectional=true, numLayers=config["Nlayers"], lstminit...)
    else
        p2=p1
        enc2 = enc1
    end

    emb1 = load_embed(vocab, config, embeddings; winit=randn)
    if config["seperate_emb"]
        emb2 = load_embed(vocab, config, embeddings; winit=randn)
    else
        emb2=emb1
    end
    attentions = (p1,p2)
    if config["self_attention"]
        attentions=(attentions...,
                    PositionalAttention(memory=config["H"],query=config["H"], att=config["attdim"],normalize=true))
    end
    if config["copy"]
        copygate  = Dense(input=config["H"], output=length(attentions), activation=NonAct())
        copygate2 = Dense(input=config["H"], output=1, activation=Sigm())
    else
        copygate = nothing
    end


    Recombine{T}(emb1,
                 emb2,
                 LSTM(;input=dinput, hidden=config["H"], numLayers=config["Nlayers"],lstminit...),
                 Linear(;input=config["H"]+length(attentions)*config["attdim"], output=config["H"], winit=linear_init(config["H"]+2config["attdim"]), binit=linear_init(config["H"]+2config["attdim"])),
                 Linear(;input=config["H"], output=length(vocab), winit=myinit, binit=myinit),
                 copygate,
                 copygate2,
                 Multiply(;input=config["E"], output=config["Z"], winit=torchinit),
                 enc1,
                 enc2,
                 PositionalAttention(memory=2config["H"], query=2config["H"], att=config["H"], normalize=true),
                 Linear(;input=2config["H"], output=2config["Z"], winit=linear_init(2config["H"]), binit=linear_init(2config["H"])),
                 zeroarray(arrtype,config["H"],config["Nlayers"]),
                 zeroarray(arrtype,config["H"],config["Nlayers"]),
                 attentions,
                 Pw{Float32}(2config["Z"], config["Kappa"]),
                 vocab,
                 config)
end

calc_ppl(model::Recombine, data; trnlen=1) = train!(model, data; eval=true, dev=nothing, trnlen=trnlen)
calc_mi(model::Recombine, data) = nothing
calc_au(model::Recombine, data) = nothing
sampleinter(model::Recombine, data) = nothing

function preprocess(m::Recombine{T}, train, devs...) where T<:DataSet
    dist = Levenshtein()
    thresh, cond, maxcnt =  m.config["dist_thresh"], m.config["conditional"], m.config["max_cnt_nb"]
    sets = map((train,devs...)) do set
            map(d->xfield(T,d,cond),set)
    end
    sets = map(unique,sets)
    words = map(sets) do set
        map(d->String(map(UInt8,d)),set)
    end
    trnwords = first(words)
    adjlists  = []
    for (k,set) in enumerate(words)
        adj = Dict((i,[]) for i=1:length(set))
        processed = []
        for i in progress(1:length(set))
            cw            = set[i]
            neighbours    = []
            cnt   = 0
            inds  = randperm(length(trnwords))
            for j in inds
                w    = trnwords[j]
                diff = compare(cw,w,dist)
                if diff > thresh && diff != 1
                    push!(neighbours,j)
                    cnt+=1
                    cnt == maxcnt && break
                end
            end
            if isempty(neighbours)
                push!(neighbours, rand(inds))
            end
            for n in neighbours
                x′′ = []
                ntokens = sets[1][n]
                xtokens = sets[k][i]
                tokens = collect(setdiff(xtokens,ntokens))
                cw     = String(map(UInt8,tokens))
                for l in inds
                    if l != n && l !=i
                        w    = trnwords[l]
                        diff = compare(cw,w,dist)
                        if diff > 0.5
                            push!(processed, (x=xtokens, xp=ntokens, xpp=sets[1][l]))
                            cnt+=1
                            cnt == 3 && break
                        end
                    end
                end
            end
        end
        push!(adjlists, processed)
    end
    return adjlists
end

function beam_decode(model::Recombine, x, xp, xpp, z; forviz=false)
    vocab = model.vocab
    T,B,H,V,E = eltype(arrtype), size(z,2), model.config["H"], length(vocab.tokens), model.config["E"]
    protos  = (xp=xp,xpp=xpp)
    masks   = (arrtype(xp.mask'*T(-1e18)),  arrtype(xpp.mask'*T(-1e18)))
    states  = (_repeat(model.h0,B,dim=2), _repeat(model.c0,B,dim=2))
    context = ntuple(i->zeroarray(arrtype,model.config["H"],B), 1)
    projs   = (arrtype(copy_projection(vocab, xp.tokens)),
               arrtype(copy_projection(vocab, xpp.tokens)))
    #outhiddens = Any[states[1]]
    limit   = model.config["maxLength"]
    preds   = ones(Int,B,limit+1) .* specialIndicies.bow
    traces  = [(zeros(1, B), states, copy(preds), preds[:,1], nothing, nothing, context, states[1])]
    protos  = (xp=xp,xpp=xpp)
    for i=1:limit
        traces = beam_search(model, traces, protos, masks, projs, z; step=i, forviz=forviz)
    end
    outputs     = map(t->t[3][:,2:end], traces)
    probs       = map(t->exp.(vec(t[1])), traces)
    score_arr, output_arr = traces[1][5], traces[1][6]
    return outputs, probs, score_arr, output_arr
end


function beam_search(model::Recombine, traces, protos, masks, projs, z; step=1, forviz=false)
    result_traces = []
    bw = model.config["beam_width"]
    Kpos = model.config["Kpos"]
    positions = (nothing,nothing)
    @inbounds for i=1:length(traces)
        probs, states, preds, cinput, scores_arr, output_arr, context, hiddenseq = traces[i]
        if model.config["positional"]
            positions = map(p->position_calc(size(p.tokens,2), step; K=Kpos), protos)
        end
        if model.config["self_attention"]
            self_proj = arrtype(copy_projection(model.vocab, preds[:,1:step]))
            self_mask = (preds[:,1:step] .== specialIndicies.mask)
            self_mask[:,1] .= true
            self_mask = arrtype(eltype(arrtype)(-1e18) .* self_mask')  # T' x B
        else
            self_proj, self_mask = nothing, nothing
        end
        output, states, _, weights, context = decode_onestep(model, states, protos, masks, projs, z, cinput, context, positions, hiddenseq, self_proj, self_mask)
        output = output|>cpucopy
        cur_probs  = exp.(output)
        step == 1 ? negativemask!(output,1:6) : negativemask!(output,1:2,4,7)
        srtinds = mapslices(x->sortperm(x; rev=true), output, dims=1)[1:bw,:]
        cprobs = sort(output; rev=true, dims=1)[1:bw,:]
        inputs = srtinds
        if step > 1
            stopped = findall(p->p==specialIndicies.eow, preds[:,step])
            if step > 2; cprobs[1,stopped] .= 0.0; end
            cprobs[2:bw,stopped] .= -100
            inputs[:,stopped] .= specialIndicies.eow
        end
        scores_softmaxed = forviz ? map(Array, weights) : nothing
        push!(result_traces, ((cprobs .+ probs), states, preds, inputs, scores_softmaxed, cur_probs, scores_arr, output_arr, context, hiddenseq))
    end
    global_probs     = mapreduce(first, vcat, result_traces)
    global_srt_inds  = mapslices(x->sortperm(x; rev=true), global_probs, dims=1)[1:bw,:]
    global_srt_probs = sort(global_probs; rev=true, dims=1)[1:bw,:]
    new_traces = []
    @inbounds for i=1:bw
        probs      = global_srt_probs[i:i,:]
        inds       = map(s->divrem(s,bw), global_srt_inds[i,:] .- 1)
        states     = ntuple(k->cat((result_traces[trace+1][2][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds))..., dims=2),2)
        if model.config["self_attention"]
            oldhiddenseq  = cat((result_traces[trace+1][10][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds))..., dims=2)
            hiddenseq     = cat(oldhiddenseq,states[1],dims=3) # get the last layer
        else
            hiddenseq  = nothing
        end
        context    = ntuple(k->cat((result_traces[trace+1][9][k][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))..., dims=2),1)
        inputs     = vcat((result_traces[trace+1][4][loc+1,bi] for (bi, (trace, loc)) in enumerate(inds))...)
        if step == 1
            old_preds  = copy(result_traces[1][3])
        else
            old_preds  = vcat((result_traces[trace+1][3][bi:bi,:] for (bi, (trace, _)) in enumerate(inds))...)
        end
        old_preds[:,step+1] .= inputs

        if forviz
            scores  = ntuple(k->hcat((result_traces[trace+1][5][k][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...),3)
            scores  = map(s->expand(s,dim=3),scores)
            outsoft = hcat((result_traces[trace+1][6][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...)
            outsoft = expand(outsoft,dim=3)
            if step == 1
                scores_arr    = scores
                output_arr    = outsoft
            else
                old_scores    = ntuple(k->cat([result_traces[trace+1][7][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2),3)
                old_outputs   = cat([result_traces[trace+1][8][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2)
                scores_arr    = ntuple(i->cat(old_scores[i],scores[i],dims=3),2) #
                if model.config["self_attention"]

                else

                end
                output_arr    = cat(old_outputs,outsoft,dims=3)
            end
        else
            scores_arr, output_arr = nothing, nothing
        end
        push!(new_traces, (probs,states,old_preds,old_preds[:,step+1], scores_arr, output_arr, context, hiddenseq))
    end
    return new_traces
end



function decode(model::Recombine, x, xp, xpp, z; sampler=argmax, training=true, mixsampler=false, temp=0.2, cond=false)
    vocab = model.vocab
    T,B,H,V,E = eltype(arrtype), size(z,2), model.config["H"], length(vocab.tokens), model.config["E"]
    protos  = (xp=xp,xpp=xpp)
    masks   = (arrtype(xp.mask'*T(-1e18)),  arrtype(xpp.mask'*T(-1e18)))
    states  = (_repeat(model.h0,B,dim=2), _repeat(model.c0,B,dim=2))
    context = ntuple(i->zeroarray(arrtype,model.config["H"],B), 1)
    projs   = (arrtype(copy_projection(vocab, xp.tokens)),
               arrtype(copy_projection(vocab, xpp.tokens)))
    positions = (nothing,nothing)
    limit   = training ? size(x,2)-1  : model.config["maxLength"]
    preds   = training ? copy(x) : ones(Int,B,limit+1) .* specialIndicies.bow
    probs   = ones(B)
    outputs, copyoutputs = [], []
    outhiddens = Any[zeroarray(arrtype,size(states[1]))]
    is_input_generated = falses(B)
    is_finished = falses(B)
    for i=1:limit
         if model.config["positional"]
             positions = map(p->position_calc(size(p.tokens,2), i; K=model.config["Kpos"]), protos)
         end
         if model.config["self_attention"]
             self_proj = arrtype(copy_projection(vocab, preds[:,1:i]))
             self_mask = (preds[:,1:i] .== specialIndicies.mask)
             self_mask[:,1] .= true
            # self_mask[:,end] .= (self_mask[:,end] .| is_input_generated .| is_finished)
             self_mask = arrtype(T(-1e18) .* self_mask')
             H,B = size(states[1])
             hiddenseq = reshape(cat1d(outhiddens...),H,B,:)
         else
             hiddenseq, self_proj, self_mask = nothing, nothing, nothing
         end
         output,states,_,_,context, copyprobs = decode_onestep(model, states, protos, masks, projs, z, preds[:,i], context, positions, hiddenseq, self_proj, self_mask)
         push!(outputs, output)
         # if model.config["copy"]
         #     push!(copyoutputs, copyprobs)
         # end

         if !training
             output = cpucopy(output)
             i == 1 ? negativemask!(output,1:7) : negativemask!(output,1:2,4,7)
             if mixsampler
                 tempout = softmax(output ./ temp, dims=1)
                 s1 = vec(mapslices(catsample, tempout, dims=1))
                 s2 = vec(mapslices(argmax, output, dims=1))
                 preds[:,i+1] = [(gen ? s2[k] : s1[k])  for (k,gen) in enumerate(is_input_generated)]
             else
                 preds[:,i+1] = vec(mapslices(sampler, softmax(output,dims=1), dims=1))
             end
             if cond
                 pred = preds[:,i+1]
                 preds[:,i+1] = [(gen ? pred[k] : x[k,i+1])  for (k,gen) in enumerate(is_input_generated)]
                 probs += (output[findindices(output,preds[:,i+1])] .* .!is_finished .* is_input_generated)
             else
                 probs += (output[findindices(output,preds[:,i+1])] .* .!is_finished)
             end
             # is_finished .= (is_finished .| (preds[:,i+1] .== specialIndicies.eow))
             # is_input_generated .= (is_input_generated .| (preds[:,i+1] .== specialIndicies.sep))
         end
         is_finished .= (is_finished .| (preds[:,i+1] .== specialIndicies.eow))
         is_input_generated .= (is_input_generated .| (preds[:,i+1] .== specialIndicies.sep))
         if model.config["self_attention"]
             push!(outhiddens, states[1])
         end
    end
    # if model.config["copy"]
    #     copyoutputs = reshape(cat1d(copyoutputs...), V, B, limit)
    # end
    reshape(cat1d(outputs...), V, B, limit), preds[:,2:end], exp.(probs), copyoutputs
end


function decode_onestep(model::Recombine, states, protos, masks, projs, z, input, prev_context, positions=nothing, hiddens=nothing, self_proj=nothing, self_mask=nothing)
    pdrop, attdrop, outdrop = model.config["pdrop"], model.config["attdrop"], model.config["outdrop"]
    e  = dropout(model.decembed(input),pdrop)
    if get(model.config,"feedcontext",false)
        xi = vcat(e, z, prev_context[1])
    else
        xi = vcat(e, z)
    end
    out = model.decoder(xi, states[1], states[2]; hy=true, cy=true)
    hquery = out.y
    #hquery  = vcat(out.y,xi)
    xp_attn, _, xp_weight = model.attentions[1](protos.xp.context, hquery; mask=masks[1], pdrop=attdrop, positions=positions[1]) #positions[1]
    xpp_attn, _, xpp_weight = model.attentions[2](protos.xpp.context, hquery; mask=masks[2], pdrop=attdrop, positions=positions[2])
    attns, weights = (xp_attn,xpp_attn), (xp_weight,xpp_weight)
    if model.config["self_attention"]
        self_attn, _, self_weight = model.attentions[3](hiddens, hquery; mask=self_mask)
        attns, weights, projs = (attns...,self_attn),  (weights...,self_weight), (projs...,self_proj)
    end
    ydrop = out.y #model.config["outdrop_test"] ? dropout(out.y, outdrop; drop=true) : dropout(out.y, outdrop)
    comb_features = dropout(model.combine(vcat(ydrop,attns...)),outdrop)
    ypred = model.output(comb_features)
    copy = specialIndicies.copy
    if model.config["copy"]
        pred_probs = softmax(ypred, dims=1)
        dists  = weights
        seq_probs  = softmax(model.copygate(hquery),dims=1)
        copy_weights = model.copygate2(hquery)#pred_probs[copy:copy,:] # 1 x B
        weighted_dists = [dists[i] .* seq_probs[i:i, :]  for i=1:length(dists)]
        copy_probs = sum([bmm(projs[i], expand(weighted_dists[i],dim=2)) for i=1:length(dists)]) .+ EPS # V X T X B TIMES T X 1 X B
        copy_probs = reshape(copy_probs,size(ypred))
        comb_probs = log.(copy_probs .* (copy_weights .+ EPS) .+ ((1+EPS) .- copy_weights) .* pred_probs)
        pred_logits = comb_probs
    else
        pred_logits  = logp(ypred, dims=1)
        copy_weights = exp.(pred_logits[copy:copy,:])
        copy_probs   = nothing
    end
    pred_logits, (out.hidden, out.memory), attns, weights, (comb_features,), copy_probs
end


function encode(m::Recombine, x, xp, xpp; prior=false)
    pdrop =  m.config["pdrop"]
    xp_context    = m.xp_encoder(dropout(m.encembed(xp.tokens), pdrop)).y
    xpp_context   = m.xp_encoder(dropout(m.encembed(xpp.tokens),pdrop)).y
    if !m.config["kill_edit"]
        if prior
            μ = zeroarray(arrtype,2latentsize(m),size(x.tokens,1))
        else
            x_context = m.x_encoder(m.encembed(x.tokens[:,2:end])).y
            x_embed   = source_hidden(x_context,x.lens)
            xp_embed  = source_hidden(xp_context,xp.lens)
            xpp_embed = source_hidden(xpp_context,xpp.lens)
            μ = m.z_emb(vcat(m.x_xp_inter(xp_embed,x_embed;feature=true)[1],
                             m.x_xp_inter(xpp_embed,x_embed;feature=true)[1])
                        )
        end
        z = sample_vMF(μ, m.config["eps"], m.config["max_norm"], m.pw; prior=prior)
    else
        z = zeroarray(arrtype,2latentsize(m),xp.batchSizes[1])
    end
    return z, (xp...,context=xp_context), (xpp...,context=xpp_context)
end


function ind2BT(write_inds, copy_inds, H, BT)
    write_batches  = map(ind->(ind ÷ H) + 1, write_inds .- 1)
    copy_batches   = map(ind->(ind ÷ H) + 1, copy_inds .- 1)
    inds = [[write_inds[findall(b->b==i, write_batches)]; copy_inds[findall(b->b==i, copy_batches)]] for i=1:BT]
    inds_masked, inds_mask  = PadSequenceArray(inds, pad=specialIndicies.mask, makefalse=false)
    return (tokens=inds_masked, mask=inds_mask, sumind=findall(length.(inds) .> 0), unpadded=inds)
end

function condmask!(ytokens, inds, B)
    unpadded = reshape(inds.unpadded, B, :)
    for i=1:B
        maskind = 1:findfirst(x->x==specialIndicies.sep,ytokens[i,:])
        unpadded[i,maskind] = map(i->Int[],maskind)  # FIXME: it is completely wrong
    end
    unpadded = reshape(unpadded,:)
    (PadSequenceArray(unpadded, pad=specialIndicies.mask, makefalse=false)...,
     sumind=findall(length.(unpadded) .> 0),
     unpadded=unpadded)
end

function loss(model::Recombine, data; average=false, eval=false, prior=false, cond=false, training=true)
    x, xp, xpp,_, unbatched = data
    B = length(first(unbatched))
    z, Txp, Txpp = encode(model, x, xp, xpp; prior=false)
    output, preds,_,copylogit = decode(model, x.tokens, Txp, Txpp, z; cond=cond, training=training)
    preds = [trimencoded(preds[i,:], eos=true) for i=1:B]
    # if !training && cond
    #     return output, preds
    # end
    ytokens, ymask = x.tokens[:,2:end], x.mask[:, 2:end]
    if !eval
        #linds = KnetLayers.findindices(output, ytokens .* ymask)
        loss = nllmask(output, ytokens .* ymask; average=true)
    else
        xinds = ytokens .* ymask
        loss = []
        for i=1:B
            inds = fill!(similar(xinds),0)
            inds[i,:] .= xinds[i,:]
            linds = KnetLayers.findindices(output, inds)
            push!(loss,-sum(output[linds]))
        end
    end
    if !training
        return loss, preds
    else
        return loss
    end
end

function copy_indices(xp, xmasked, L::Integer, offset::Integer=0)
    indices = Int[]
    B,T = size(xmasked)
    for t=2:T
        @inbounds for i=1:B
            start = (t-2) * L * B + (i-1)*L + offset
            token =  xmasked[i,t]
            if token != specialIndicies.mask
                @inbounds for i in findall(t->t==token,xp[i])
                    push!(indices, start+i)
                end
            end
        end
    end
    return indices
end

function getbatch(model::Recombine, iter, B)
    edata = collect(Iterators.take(iter,B))
    b = length(edata); b==0 && return nothing
    unk, mask, eow, bow, sep = specialIndicies
    maxL = model.config["maxLength"]
    V = length(model.vocab.tokens)
    d           = (x, xp1, xp2) = unzip(edata)
    xp,xpp      = (xp1,xp2) #rand()>0.5 ? (xp1,xp2) : (xp2,xp1)
    x           = map(s->[bow;s;eow],limit_seq_length(x;maxL=maxL))
    xp          = map(s->[s;eow], limit_seq_length(xp;maxL=maxL))  # FIXME: CHANGED
    xpp         = map(s->[s;eow], limit_seq_length(xpp;maxL=maxL)) # FIXME: CHANGED
    pxp         = PadSequenceArray(xp, pad=mask, makefalse=true)
    pxpp        = PadSequenceArray(xpp, pad=mask, makefalse=true)
    px          = PadSequenceArray(x, pad=mask, makefalse=false)
    seq_xp      = (pxp...,lens=length.(xp))
    seq_xpp     = (pxpp...,lens=length.(xpp))
    seq_x       = (px...,lens=length.(x) .- 1)
    Tp, Tpp     = size(seq_xp.tokens,2),  size(seq_xpp.tokens,2)
    L = V + Tp + Tpp
    xp_copymask  = copy_indices(xp, seq_x.tokens,  L, V)
    xpp_copymask = copy_indices(xpp, seq_x.tokens, L, V+Tp)
    return seq_x, seq_xp, seq_xpp, (xp_copymask, xpp_copymask), d
end

function train!(model::Recombine, data; eval=false, dev=nothing, returnlist=false, trnlen=1)
    ppl = typemax(Float64)
    if !eval
        bestparams = deepcopy(parameters(model))
        setoptim!(model,model.config["optim"])
        model.config["rpatiance"]  = model.config["patiance"]
        model.config["rwritedrop"] = model.config["writedrop"]
        #model.config["gradnorm"]   = 0.0
    else
        data, evalinds = create_ppl_examples(data, 1000)
        losses = []
    end
    total_iter, lss, ntokens, ninstances =0, 0.0, 0.0, 0.0

    for i=1:(eval ? 1 : model.config["epoch"])
        lss, ntokens, ninstances = 0.0, 0.0, 0.0
        dt  = Iterators.Stateful((eval ? data : shuffle(data)))
        msg(p) = string(@sprintf("Iter: %d,Lss(ptok): %.2f,Lss(pinst): %.2f, PPL(test): %.2f", total_iter, lss/ntokens, lss/ninstances, ppl))
        for i=1:(((length(dt)-1) ÷ model.config["B"])+1)
            total_iter += 1
            d = getbatch(model, dt, model.config["B"])
            isnothing(d) && continue
            if !eval
                J = @diff loss(model, d)
                # if total_iter < 50
                #     model.config["gradnorm"] = max(model.config["gradnorm"], 2*clip_gradient_norm_(J, Inf))
                # else
                if model.config["gradnorm"] > 0
                    clip_gradient_norm_(J, model.config["gradnorm"])
                end
                # end
                for w in parameters(J)
                    g = grad(J,w)
                    if !isnothing(g)
                        KnetLayers.update!(value(w), g, w.opt)
                    end
                end
                #model.config["rwritedrop"] = max(model.config["rwritedrop"] - 0.0001,model.config["writedrop"])
            else
                #J = loss(model, d; eval=true)
                ls = loss(model, d; eval=true)
                append!(losses,ls)
                J = mean(ls)
            end
            b = size(d[1].mask,1)
            n = sum(d[1].mask[:,2:end])
            lss += (value(J)*b)
            ntokens += n
            ninstances += b
            if !eval && i%500==0
                #print_ex_samples(model, data; beam=true)
                if !isnothing(dev)
                     print_ex_samples(model, dev; beam=true)
                     #calc_ppl(model, dev)
                end
            end
        end
        if !isnothing(dev)
            newppl = calc_ppl(model, dev; trnlen=trnlen)[1]
            if newppl > ppl-0.0025 # not a good improvement
                #lrdecay!(model, model.config["lrdecay"])
                model.config["rpatiance"] = model.config["rpatiance"] - 1
                println("patiance decay, rpatiance: $(model.config["rpatiance"])")
                if model.config["rpatiance"] == 0
                    lrdecay!(model, model.config["lrdecay"])
                    model.config["rpatiance"] = model.config["patiance"]
                    #break
                end
            else
                model.config["rpatiance"] = model.config["patiance"]
            end

            if newppl < ppl
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
                s_losses, kl_calc(model), log(trnlen), data, eval_inds
            end
        end
#        total_iter > 400000 && break
    end

    # if !isnothing(dev) && !eval
    #     for (best, current) in zip(bestparams,parameters(model))
    #         copyto!(value(current),value(best))
    #     end
    # end
    return (ppl=ppl, ptokloss=lss/ntokens, pinstloss=lss/ninstances)
end

function condeval(model::Recombine, data; k=3)
    data, evalinds = create_ppl_examples(data, length(data))
    losses, preds = [], []
    dt = Iterators.Stateful(data)
    for i in progress(1:((length(data)-1) ÷ model.config["B"])+1)
        d = getbatch(model, dt, model.config["B"])
        isnothing(d) && continue
        _, pred = loss(model, d; eval=true, cond=true, training=false)
        append!(preds,pred)
    end
    data_r = []
    for (k,inds) in enumerate(evalinds)
        protos = map(d->(xp=d.xp,xpp=d.xpp),data[inds])
        for x in unique(preds[inds])
            for proto in protos
                push!(data_r, (x=x,proto...))
            end
        end
    end
    data_r, evalinds_r = create_ppl_examples(data_r, length(data_r))
    dt = Iterators.Stateful(data_r)
    for i in progress(1:((length(data_r)-1) ÷ model.config["B"])+1)
        d = getbatch(model, dt, model.config["B"])
        isnothing(d) && continue
        lss = loss(model, d; eval=true, cond=true, training=true)
        append!(losses,lss)
    end
    s_losses = []
    result = Dict()
    for inds in evalinds_r
        lss = -logsumexp(-losses[inds])
        xy  = data_r[first(inds)].x
        x,y = split_array(xy, specialIndicies.sep)
        push!(get!(result,x,[]),(y,lss))
    end
    exact_match = .0
    top_k_match = .0
    tp, fp, fn = .0, .0, .0
    total = .0
    for xy in unique(map(x->x.x,data))
        x,ref = split_array(xy, specialIndicies.sep)
        ref = [ref;specialIndicies.eow]
        results = result[x]
        r = sortperm(results, by=x->x[2])
        pred_here = results[r[1]][1]
        @show pred_here
        @show ref
        tp += length([p for p in pred_here if p ∈ ref])
        fp += length([p for p in pred_here if p ∉ ref])
        fn += length([p for p in ref if p ∉ pred_here])
        exact_match += (pred_here == ref)
        top_k_match += (ref ∈ first.(results[r[1:min(k,end)]]))
        total += 1
    end
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)
    if prec == 0 || rec == 0
        f1 = 0
    else
        f1 = 2 * prec * rec / (prec + rec)
    end
    (exact=exact_match/total, top_k=top_k_match/total, f1=f1)
end

function vmfKL(m::Recombine)
   k, d = m.config["Kappa"], 2m.config["Z"]
   k*((besseli(d/2.0+1.0,k) + besseli(d/2.0,k)*d/(2.0*k))/besseli(d/2.0, k) - d/(2.0*k)) +
   d * log(k)/2.0 - log(besseli(d/2.0,k)) -
   loggamma(d/2+1) - d * log(2)/2
end

function to_json(parser, fname)
    lines = readlines(fname)
    data = map(line->parseDataLine(line, parser), readlines(fname))
    json = map(d->JSON.lower((inp=d.input,out=d.output)), data)
    fout = fname[1:first(findfirst(".", fname))-1] * ".json"
    open(fout, "w+") do f
        JSON.print(f,json, 4)
    end
    return fout
end

using Plots
if get(ENV,"recombine_viz",false)
    pyplot()
end


function viz(model, data; N=5)
    samples = sample(model, shuffle(data); N=N, sampler=sample, prior=false, beam=true, forviz=true)
    vocab = model.vocab.tokens
    #json = map(d->JSON.lower((x=vocab[d.x], xp=vocab[d.xp], xpp=vocab[d.xpp], scores1=d.scores[1], scores2=d.scores[2], probs=d.probs)), samples)
    #open("attention_maps.json", "w+") do f

    #    JSON.print(f,json, 4)
    #end
    for i=1:length(samples)
        x, scores, probs, xp, xpp = samples[i]
        attension_visualize(model.vocab, probs, scores, xp, xpp, x; prefix="$i")
        println("samples: ", join(model.vocab.tokens[x],' '))
        println("xp: ", join(model.vocab.tokens[xp],' '))
        println("xpp: ", join(model.vocab.tokens[xpp],' '))
        println("----------------------")
    end
end
function removeunicodes(toElement)
    vocab = copy(toElement)
    vocab[1:4] = ["<unk>", "<mask>", "<eow>", "<bow>"]
    return vocab
end

function attension_visualize(vocab, probs, scores, xp, xpp, x; prefix="")
    x  = vocab.tokens[x]
    y1 = vocab.tokens[xp]
    y2 = vocab.tokens[xpp]
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
    words = removeunicodes(vocab.tokens.toElement)
    y3 = [words; "xp" .* string.(1:length(xp)); "xpp" .* string.(1:length(xpp))]
    l = @layout [ a{0.5w} [b
                           c{0.5h}]]

    p1 = heatmap(scores[1];yticks=(1:length(y1),y1),title="xp-x attention", attributes...)
    p2 = heatmap(scores[2];yticks=(1:length(y2),y2),title="xpp-x attention", attributes...)
    p3 = heatmap(probs;yticks=(1:length(y3),y3), title="action probs", attributes...)
    p  = Plots.plot(p3,p1,p2; layout=l)
    Plots.savefig(p, prefix*"_attention_map.pdf")
end


function pickprotos_conditional(model::Recombine{SCANDataSet}, esets; subtask=nothing)
    dist  = Levenshtein()
    trn        = map(d->xfield(SCANDataSet,d,true),esets[1])
    tst        = map(d->xfield(SCANDataSet,d,true),esets[2])
    trnwords   = map(d->d.input,esets[1])
    trnstrs    = map(d->String(map(UInt8,d.input)),esets[1])
    testwords  = map(d->d.output,esets[2])
    teststrs   = map(d->String(map(UInt8,d.input)),esets[2])
    processed  = []
    for i=1:length(testwords)
        current_word = testwords[i]
        current_str  = teststrs[i]
        xp_candidates = []
        for j in 1:length(trnstrs)
            trn_word = trnwords[j]
            trn_str  = trnstrs[j]
            sdiff = length(setdiff(current_word,trn_word))
            ldiff = compare(current_str,trn_str,dist)
            if ldiff > 0.5 && ldiff != 1.0 && sdiff > 0
                push!(xp_candidates,(j,ldiff))
            end
        end
        if isempty(xp_candidates)
            push!(xp_candidates, rand(1:length(trnwords)))
        else
            xp_candidates = first.(sort(xp_candidates, by=x->x[2], rev=true)[1:min(end,5)])
        end
        for n in xp_candidates
            xpp_candidates = []
            xp_tokens      = trnwords[n]
            diff_tokens    = setdiff(current_word,xp_tokens)
            diff_str       = String(map(UInt8,diff_tokens))
            for l=1:length(trnwords)
                if l != n
                    ldiff = compare(diff_str,trnstrs[l],dist)
                    #sdiff = length(symdiff(diff_tokens,trnwords[l]))
                    lendiff = abs(length(trnwords[l])-length(diff_tokens))
                    if ldiff > 0.5 && lendiff < 5
                        push!(xpp_candidates,(l,ldiff))
                        #push!(processed, (x=xtokens, xp=ntokens, xpp=sets[1][l]))
                        # cnt+=1
                        #cnt == 3 && break
                    end
                end
            end
            if isempty(xpp_candidates)
                current_xpp = xfield(SCANDataSet,rand(esets[1]),true)
                push!(processed, (x=tst[i], xp=trn[n], xpp=rand(trn))) #push!(neighbours, rand(inds))
            else
                xpp_candidates = first.(sort(xpp_candidates, by=x->x[2], rev=true)[1:min(end,5)])
                for xpp in xpp_candidates
                    current_xpp = xfield(SCANDataSet,trn[xpp],true)
                    push!(processed, (x=tst[i], xp=trn[n], xpp=trn[xpp]))
                end
            end
        end
    end
    processed
end


function pickprotos_conditional(model::Recombine{SIGDataSet}, processed, esets; subtask="reinflection")
    dist   = Levenshtein()
    trndata = esets[1]
    trnstr  = map(d->String(map(UInt8,d.surface)),trndata)
    data = []
    for k=2:length(esets)
        processed = []
        cur_test  = esets[k]
        for i=1:length(cur_test)
            cur_input = cur_test[i].surface
            cur_str   = map(UInt8,cur_input)
            neihgbours = []
            for j=1:length(trnstr)
                trn_ex_str = trnstr[j]
                diff = compare(cur_str,trn_ex_str,dist)
                push!(neighbours,(j,diff))
            end
            xp = trndata[findmax(second.(neighbours))[2]]
            for j=1:length(trnstr)
                xpp     = trndata[j]
                xpp_str = trnstr[j]
                difftag = setdiff(cur_input, xp.surface, xpp.surface)
                trn_ex_str = trnstr[j]
                diff = compare(cur_str,trn_ex_str,dist)
                push!(neighbours,(j,diff))
            end
        end
    end
    data
end

function preprocess(m::Recombine{SIGDataSet}, train, devs...)
    println("preprocessing SIGDataSet")
    T = SIGDataSet
    dist = Levenshtein()
    thresh, cond, maxcnt, masktags, subtask =  m.config["dist_thresh"], m.config["conditional"], m.config["max_cnt_nb"], m.config["masktags"], m.config["subtask"]
    sets = map((train,devs...)) do set
            map(d->(x=xfield(T,d,cond; masktags=masktags, subtask=subtask), lemma=d.lemma, surface=d.surface, tags=(masktags ?  fill!(similar(d.tags),specialIndicies.mask) : d.tags)),set)
    end
    sets = map(set->unique(s->s.x,set),sets)

    words = map(sets) do set
        map(d->String(map(UInt8,d.x)),set)
    end

    lemmas = map(sets) do set
        map(d->String(map(UInt8,d.lemma)),set)
    end

    tags = map(sets) do set
            map(d->String(map(UInt8,d.tags)),set)
    end

    surfaces = map(sets) do set
        map(d->String(map(UInt8,d.surface)),set)
    end

    trnwords   = first(words)
    trnlemmas  = first(lemmas)
    trntags    = first(tags)
    trnsurface = first(surfaces)

    adjlists  = []
    for (k,set) in enumerate(words)
        adj = Dict((i,[]) for i=1:length(set))
        processed = []
        lemma_set   = lemmas[k]
        tag_set     = tags[k]
        surface_set = surfaces[k]
        for i in progress(1:length(set))
            cw            = set[i]
            clemma        = lemma_set[i]
            ctag          = tag_set[i]
            csurface      = surface_set[i]
            neighbours    = []
            cnt   = 0
            inds  = randperm(length(trnwords))
            for j in inds
                w          = trnwords[j]
                lemma      = trnlemmas[j]
                tag        = trntags[j]
                surface = trnsurface[j]
                diff  = compare(cw,w,dist)
                ldiff = compare(clemma, lemma, dist)
                #tdiff = compare(ctag,tag,dist)
                sdiff = compare(csurface,surface,dist)
                if  diff != 1 && (ldiff > thresh  || sdiff > thresh)
                    push!(neighbours,j)
                    cnt+=1
                    cnt == maxcnt && break
                end
            end
            if isempty(neighbours)
                push!(neighbours, rand(inds))
            end
            for n in neighbours
                x′′ = []
                ntokens = sets[1][n].x
                xtokens = sets[k][i].x
                tokens  = setdiff(xtokens,ntokens)
                diffw   = String(map(UInt8,tokens))
                xtag    = sets[k][i].tags
                ntag    = sets[1][n].tags
                difftag = setdiff(xtag,ntag)
                xtag    = tags[k][i]
                cnt = 0
                for l in inds
                    if l != n && l !=i
                        w           = trnwords[l]
                        tag         = trntags[l]
                        tag_tokens  = sets[1][l].tags
                        diff        = compare(diffw,w,dist)
                        tdiff       = compare(trntags[n],tag,dist)
                        trealdiff   = setdiff(difftag,tag_tokens)
                        if length(trealdiff) == 0 &&  tdiff > thresh
                            push!(processed, (x=xtokens, xp=ntokens, xpp=sets[1][l].x))
                            cnt+=1
                            cnt == maxcnt && break
                        end
                    end
                end
            end
        end
        push!(adjlists, processed)
    end
    return adjlists
end



function pickprotos(model::Recombine{SCANDataSet}, processed, esets; subtask=nothing)
    p_std, p_mean = std_mean((length(p.xp) for p in first(processed)))
    pthresh = p_mean
    pp_std, pp_mean = std_mean((length(p.xpp) for p in first(processed)))
    ppthresh = pp_mean
    @show p_mean, pp_mean
    eset    = length(esets) == 3  ? [esets[1];esets[3]] : esets[1]
    set     = map(d->xfield(model.config["task"],d,model.config["conditional"]),eset)
    set     = unique(set)
    inputs  = map(d->d.input, eset)
    outputs = map(d->d.output, eset)
    data    = []
    inpdict, outdict =  Set(Int[]), Set(Int[])
    for d in inputs; for t in d; push!(inpdict,t); end; end
    for d in outputs; for t in d; push!(outdict,t); end; end
    for i=1:length(set)
        length(set[i]) > pthresh && continue
         for j=1:length(set)
            j==i && continue
            length(set[j]) > ppthresh && continue
            push!(data, (x=Int[specialIndicies.bow], xp=set[i], xpp=set[j]))
        end
    end
    data, Set(inputs), Set(outputs), inpdict, outdict, nothing, nothing
end

function pickprotos(model::Recombine{SIGDataSet}, processed, esets; subtask="analyses")
    eset    = esets[1] #length(esets) == 3  ? [esets[1];esets[3]] : esets[1]
    vocab   = model.vocab.tokens
    set     = map(d->xfield(SIGDataSet,d,true; subtask=subtask),eset)
    outputs = map(d->[d.lemma;d.tags], eset)
    inputs  = map(d->d.surface, eset)
    tags    = map(d->d.tags, eset)

    if subtask == "reinflection"
        inputs, outputs = outputs, inputs
    end

    dicts = ntuple(i->Dict(),3)
    for (i,tset) in enumerate((inputs,outputs,tags))
        dict = dicts[i];
        for d in tset;
            for t in d;
                dict[t] = get(dict,t,0) + 1;
            end;
        end
    end

    tag_counts = dicts[3]

    dicts = map(d->Set(collect(keys(d))), dicts)

   @show tag_counts

   rare = [t for (t,cnt) in tag_counts if cnt > 1 && cnt <= 40 && isuppercase(vocab[t][1])]
   println(vocab[rare])

   weird_items = [
       i for (i, tag) in enumerate(tags)
       if length(intersect(Set(tag), Set(rare))) > 0
   ]

   # println("weird")
   # for i in weird_items
   #     inp, out = inputs[i], outputs[i]
   #     println(join(vocab[inp]," "))
   #     println(join(vocab[out]," "))
   # end

   data = []
   trnitems = 1:length(set)
   for j=1:length(esets[3])
       allow_exact = j % 2 == 1
       if length(weird_items) == 0
           push!(data, (x=Int[specialIndicies.bow], xp=set[trnitems], xpp=set[trnitems]))
           continue
       end
       tag = rand(rare)
       println(vocab[tag])
       #println(vocab[tag])
       shuffle!(weird_items)
       i1 = rand([i for i in weird_items if tag in tags[i]])
       #println(join(vocab[set[i1]]," "))
       item1, item1_out = set[i1], tags[i1]
       sort_key = i-> (!(allow_exact && tags[i] == item1_out),
                         -length(intersect(Set(tags[i]),Set(item1_out))))

       weird_available = sort([i for i in trnitems if i != i1], by=sort_key)
       i2 = weird_available[1]
       item2 = set[i2]
       item2_out = tags[i2]

       all_available = sort([i for i in trnitems if i != i1 && i != i2], by=sort_key)

       i3 = all_available[1]
       item3 = set[i3]
       for (k1,k2) in  ((i1, i2), (i1, i3), (i2, i1), (i3, i1))
           push!(data, (x=Int[specialIndicies.bow], xp=set[k1], xpp=set[k2]))
       end
   end
    #
    # for i=1:length(set)
    #      p_inp, p_out = inputs[i], outputs[i]
    #      p_lemmaptags = subtask == "reinflection" ? p_inp : p_out
    #      p_lemma, p_ts = split_array(p_lemmaptags,specialIndicies.iosep) #p_inp[1:indexsep-1],p_inp[indexsep+1:end]
    #      for j=1:length(set)
    #         j==i && continue
    #         pp_inp, pp_out = inputs[j], outputs[j]
    #         pp_lemmaptags = subtask == "reinflection" ? pp_inp : pp_out
    #         pp_lemma, pp_ts = split_array(pp_lemmaptags,specialIndicies.iosep)
    #         if 0 < length(setdiff(p_ts,pp_ts)) < 3
    #             push!(data, (x=Int[specialIndicies.bow], xp=set[i], xpp=set[j]))
    #         end
    #     end
    # end
    data, Set(inputs), Set(outputs), dicts[1], dicts[2], Set(tags), dicts[3]
end

function io_to_line(vocab::Vocabulary{SCANDataSet}, input, output; subtask=nothing)
    v = vocab.tokens; "IN: "*join(v[input],' ')*" OUT: "*join(v[output],' ')
end


function io_to_line(vocab::Vocabulary{SIGDataSet}, input, output; subtask="reinflection")
    v = vocab.tokens
    input, output = v[input], v[output]
    if subtask == "analyses"
        lemma, ts = split_array(output, isuppercaseornumeric; include=true)
        "$(join(lemma))\t$(join(input))\t$(join(ts,';'))"
    else
        lemma, ts = split_array(input, isuppercaseornumeric; include=true)
        "$(join(lemma))\t$(join(output))\t$(join(ts,';'))"
    end
end



function print_samples(model, processed, esets; beam=true, fname="samples.txt", N=400, Nsample=N, K=300, mixsampler=false)
    vocab, task, subtask  = model.vocab, model.config["task"], get(model.config, "subtask", nothing)
    nonexisttag, iosep = true, specialIndicies.sep
    printed, aug_io, probs = String[], [], []
    data, inputs, outputs, inpdict, outdict, tags, tagdict = pickprotos(model, processed, esets; subtask=subtask)
    #data = [processed[2];processed[3]] # when you want to cheat!
    iter = Iterators.Stateful(Iterators.cycle(shuffle(data)))
    while length(printed) < Nsample
        for s in sample(model, iter; N=128, prior=true, beam=beam, mixsampler=mixsampler)
            tokens = s.sampleenc
            if iosep ∈ tokens && length(findall(d->d==iosep, tokens)) == 1
                input, output = split_array(tokens, iosep)
                (length(input) == 0 || length(output)==0) && continue
                if task == SIGDataSet
                    lemmaptags = subtask == "reinflection" ? input : output
                    charseq = vocab.tokens[lemmaptags]
                    if any(map(isuppercaseornumeric, charseq))
                        lemma, tag = split_array(charseq, isuppercaseornumeric; include=true)
                        lemma, tag = vocab.tokens[lemma], vocab.tokens[tag]
                    else
                        continue
                    end
                    # !(specialIndicies.iosep ∈ lemmaptags && length(findall(d->d==specialIndicies.iosep, tokens)) == 1) && continue
                    # lemma, tag = split_array(lemmaptags, specialIndicies.iosep)
                    #nonexisttag =  tag ∉ tags && all(i->haskey(tagdict,i), tag) && length(tag) > 1 && (length(tag) == length(unique(tag)))
                    nonexisttag =  all(i->haskey(tagdict,i), tag) && length(tag) > 1 && (length(tag) == length(unique(tag)))
                    #nonexisttag = true
                end
                line = io_to_line(vocab, input, output; subtask=subtask)
                if line ∉ printed &&
                   nonexisttag &&
                   input ∉ inputs &&
                   output ∉ outputs &&
                   all(i->haskey(inpdict,i), input) &&
                   all(i->haskey(outdict,i), output)
                   push!(printed,line)
                   #@show line
                   push!(aug_io, (input=vocab.tokens[input], output=vocab.tokens[output]))
                   push!(probs, s.probs[1])
                   length(printed) == Nsample && break
                end
            end
        end
    end
    r = sortperm(probs; rev=true)[1:N]
    open(fname, "w+") do f
        for i in r; println(f,printed[i]); end
    end
    if subtask == "reinflection" # FIXME: assuming that the conditional task is always "analyses"
         aug_io = map(d->(input=d.output, output=d.input),aug_io)
    end
    return printed[r], aug_io[r]
end

function process_for_viz(vocab, pred, xp, xpp, scores, probs)
    xtrimmed    = trimencoded(pred)
    xps         = (trimencoded(xp),trimencoded(xpp))
    attscores   = [score[1:length(xps[k]),1:length(xtrimmed)] for  (k,score) in enumerate(scores)]
    ixp_end     = length(vocab)+length(xps[1])
    ixpp_start  = length(vocab)+length(xp)+1
    ixpp_end    = length(vocab)+length(xp)+length(xps[2])
    indices     = [collect(1:ixp_end); collect(ixpp_start:ixpp_end)]
    outsoft     = probs[indices,1:length(xtrimmed)]
    (x=xtrimmed, scores=attscores, probs=outsoft, xp=xps[1], xpp=xps[2])
end


function sample(model::Recombine, dataloader; N::Int=model.config["N"], sampler=argmax, prior=true, beam=true, forviz=false, mixsampler=false)
    if !(dataloader isa Base.Iterators.Stateful)
        dataloader = Iterators.Stateful(dataloader)
    end
    B  = max(model.config["B"],128)
    vocab = model.vocab
    samples = []
    while true
        d = getbatch(model, dataloader, B)
        if isnothing(d)
             @warn "sampled less variable than expected"
             break
        end
        x, xp, xpp, copymasks, unbatched = d
        b = length(first(unbatched))
        z, Txp, Txpp = encode(model, x, xp, xpp; prior=prior)
        if beam && !mixsampler # FIXME: You can do beam with mixsampler
            preds, probs, scores, outputs  = beam_decode(model, x.tokens, Txp, Txpp, z; forviz=forviz)
            if forviz
                for i=1:b
                    @inbounds push!(samples,process_for_viz(vocab,
                    preds[1][i,:],
                    xp.tokens[i,:],
                    xpp.tokens[i,:],
                    ntuple(k->scores[k][:,i,:],2),
                    outputs[:,i,:]))
                end
                length(samples) >= N && break
                continue
            end
            probs2D = hcat(probs...)
            predstr = [join(ntuple(k->trim(preds[k][i,:], vocab),length(preds)),'\n')  for i=1:b]
            predenc = [trimencoded(preds[1][i,:]) for i=1:b]
        else
            _, preds, probs = decode(model, x.tokens, Txp, Txpp, z; sampler=sampler, training=false, mixsampler=mixsampler)
            predstr  = mapslices(x->trim(x,vocab), preds, dims=2)
            probs2D  = reshape(probs,b,1)
            predenc  = [trimencoded(preds[i,:]) for i=1:b]
        end
        for i=1:b
            @inbounds push!(samples, (target = join(vocab.tokens[unbatched[1][i]],' '),
                                      xp = join(vocab.tokens[unbatched[2][i]],' '),
                                      xpp = join(vocab.tokens[unbatched[3][i]],' '),
                                      sample = predstr[i],
                                      sampleenc = predenc[i],
                                      probs = probs2D[i,:]))
        end
        length(samples) >= N && break
    end
    samples[1:N]
end

function print_ex_samples(model::Recombine, data; beam=true, mixsampler=false)
    println("generating few examples")
    #for sampler in (sample, argmax)
    for prior in (true, false)
        println("Prior: $(prior) , attend_pr: 0.0")
        for s in sample(model, data; N=10, sampler=argmax, prior=prior, beam=beam, mixsampler=mixsampler)
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


function preprocess_jacobs_format(neighboorhoods, splits, esets, edata; subtask="reinflection")
    processed = []
    for (set,inds) in zip(esets,splits)
        proc_set = []
        for (d,i) in zip(set,inds)
            !haskey(neighboorhoods,string(i-1)) && continue
            x  = xfield(SIGDataSet,d,true; subtask=subtask)
            for ns in neighboorhoods[string(i-1)]
                xp  = xfield(SIGDataSet,edata[ns[1]+1],true; subtask=subtask)
                xpp = xfield(SIGDataSet,edata[ns[2]+1],true; subtask=subtask)
                push!(proc_set, (x=x, xp=xp, xpp=xpp, ID=inserts_deletes(x,xp)))
                #push!(proc_set, (x=x, xp=xpp, xpp=xp, ID=inserts_deletes(x,xpp)))
            end
        end
        push!(processed, proc_set)
    end
    return processed
end

function read_from_jacobs_format(path, config)
    subtask = config["subtask"]
    println("reading from $path ,and subtask $(subtask)")
    hints, seed = config["hints"], config["seed"]
    fix = "hints-$hints.$seed"
    data = map(d->convert(Vector{Int},d) .+ 1, JSON.parsefile(path*"seqs.$fix.json"))
    splits = JSON.parsefile(path*"splits.$fix.json")
    neighbourhoods = JSON.parsefile(path*"neighborhoods.$fix.json")
    vocab = JSON.parsefile(path*"vocab.json")
    vocab = convert(Dict{String,Int},vocab)
    for (k,v) in vocab; vocab[k] = v+1; end
    vocab = IndexedDict(vocab)
    strdata = [split_array(vocab[d][3:end-1],"<sep>") for d in data]
    strdata = map(strdata) do  d
                lemma, tags = split_array(d[1],isuppercaseornumeric; include=true)
                (surface=d[2], lemma=lemma, tags=tags)
    end
    if isfile(path*"generated.$fix.json")
        augmented_data = map(d->convert(Vector{Int},d) .+ 1, JSON.parsefile(path*"generated.$fix.json"))
        aug = map(augmented_data) do  d
            x = vocab[d[3:end-1]]
            if length(findall(t->t=="<sep>", x)) == 1
                input, output = split_array(x,"<sep>")
                if any(map(isuppercaseornumeric, x))
                    lemma, tags   = split_array(input, isuppercaseornumeric; include=true)
                    (surface=output, lemma=lemma, tags=tags)
                    (input=output, output=input)
                else
                    nothing
                end
            else
                nothing
            end
        end
        aug  = filter!(!isnothing, aug)
    else
        aug  = []
    end
    println(length(aug))
    vocab  = Vocabulary(strdata, Parser{SIGDataSet}())
    edata  = encode(strdata,vocab)
    splits = [Int.(splits["train"]) .+ 1, Int.(splits["test_hard"]) .+ 1,  Int.(splits["val_hard"]) .+ 1]
    esets  = [edata[s] for s in splits]
    #eaug   = encode(aug,vocab)
    processed = preprocess_jacobs_format(neighbourhoods, splits, esets, edata; subtask=config["subtask"])
    model     = config["model"](vocab, config; embeddings=nothing)
    #processed  = preprocess(model, esets...)
    #save(fname, "data", processed, "esets", esets, "vocab", vocab, "embeddings", nothing)
    return processed, esets, model, aug
end
