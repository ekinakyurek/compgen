struct PositionalAttention
    key_transform::Multiply
    query_transform::Multiply
    aij_k::Embed
    aij_v::Embed
    attdim::Int
end

function PositionalAttention(;memory::Int,query::Int,att::Int, aij_k::Embed, aij_v::Embed)
    PositionalAttention(Multiply(att_weights(input=memory,output=att)),
              Multiply(att_weights(input=query,output=att)),
              aij_k,
              aij_v,
              att)
end

function single_attend(m::PositionalAttention, memory, query; pdrop=0.0)
    tkey     = m.key_transform(memory)
    tquery   = m.query_transform(query)
    score    = dropout(tquery .* tkey,pdrop) ./ sqrt(m.attdim) #[H,B]
    weights  = softmax(score;dims=1) #[H,B]
    values   = memory .* weights # [H,B]
    return (values, weights)
end

function (m::PositionalAttention)(memory, query; pdrop=0, mask=nothing, batchdim=3, positions=nothing)
    if ndims(memory)  == 2 && ndims(query) == 2
        return single_attend(m, memory,query; pdrop=0)
    end
    tquery  = m.query_transform(query) # [A,T,B]
    pscores = nothing
    if !isnothing(positions) && ndims(tquery) == 2
        pscores2D = At_mul_B(m.aij_k(positions), tquery) # A x T' times A x B
        pscores   = reshape(pscores2D, 1, size(pscores2D,1), size(pscores2D,2)) # 1 X T' X B
    elseif !isnothing(positions) && ndims(tquery) == 3
        pscores = bmm(m.aij_k(positions), permutedims(tquery, (1,3,2)); transA=true) # T' X A X T times  A X B X T  = T X T' X B
        pscores = permutedims(pscores, (3,1,2))
    end

    if  ndims(tquery) == 2
        tquery = reshape(tquery,m.attdim,1,size(tquery,2))
    end

    tkey = m.key_transform(memory) #[A,T',B]

    if isnothing(pscores)
        score = dropout(bmm(tquery,tkey,transA=true),pdrop) ./ sqrt(m.attdim) #[T,T',B]
    else
        score = dropout(bmm(tquery,tkey,transA=true) .+ pscores, pdrop) ./ sqrt(m.attdim) #[T,T',B]
    end

    if !isnothing(mask)
        score = applymask(score, mask, +)
    end
    #memory H,T',B
    weights  = softmax(score;dims=2) #[T,T',B]
    values   = mat(bmm(memory,weights;transB=true)) # [H,T,B]
    if !isnothing(pscores) && size(tquery,2) == 1
        pmemory = m.aij_v(positions) # H X T'
        values = pmemory * mat(weights) + mat(values)
    elseif !isnothing(pscores)
        pmemory = m.aij_v(positions) # A X T' X T
        pvalues = bmm(pmemory, permutedims(weights, (3,2,1)); transA=true) #T' X A X T times
        return values .+ pvalues , score, weights
    end
    return values, mat(score), mat(weights)
end

function newattention(m::PositionalAttention, memory, query; pdrop=0, mask=nothing, positions=nothing)
    tquery  = m.query_transform(query) # [A,B,T] or [A,B]
    pscores = nothing
    if !isnothing(positions) && ndims(tquery) == 2
        pscores2D = At_mul_B(m.aij_k(positions), tquery) # T' x A \times A x B
        pscores   = reshape(pscores2D,size(pscores2D,1), size(pscores2D,2), 1) # T' x B x 1
    elseif !isnothing(positions) && ndims(tquery) == 3
        pscores = bmm(m.aij_k(positions), tquery ; transA=true) # T' x A x T \times  A x B x T  = T' x B x T
    end

    if  ndims(tquery) == 2
        tquery = reshape(tquery,m.attdim,size(tquery,2), 1) #[A,B,1]
    end

    tkey = m.key_transform(memory) #[A,B,T']
    score = batch_last(bmm(batch_last(tkey), batch_last(tquery),transA=true)) #[T',B,T]
    # T' x B x T
    if !isnothing(pscores)
        score += pscores
    end

    score = dropout(score, pdrop) ./ sqrt(m.attdim)

    if !isnothing(mask)
        score = applymask2(score, reshape(mask, size(mask)..., 1), +) # size(mask) == [T', B , 1]
    end

    #memory H,B,T' # weights T' X B X T
    weights  = softmax(score;dims=1) #[T',B,T]
    values   = batch_last(bmm(batch_last(memory),batch_last(weights))) # [H,B,T]

    if !isnothing(pscores)
        if size(tquery,3) == 1
            pmemory = reshape(m.aij_v(positions), size(memory,1), length(positions), 1) # H X T' x 1
        else
            pmemory = m.aij_v(positions) # H X T' X T \times T' X B X T
        end
        values = values .+ bmm(pmemory, weights) # H X B X T times
    end

    if  size(tquery,3) == 1
        mat(values, dims=1), mat(score,dims=1), mat(weights, dims=1)
    else
        values, score, weights
    end
end


struct Recombine{Data}
    embed::Embed
    decoder::LSTM
    output::Linear
    enclinear::Multiply
    xp_encoder::LSTM
    x_encoder::LSTM
    x_xp_inter::Attention
#   x_xpp_inter::Attention
    z_emb::Linear
    h0
    c0
    xp_attention::PositionalAttention
#   xpp_attention::PositionalAttention
    pw::Pw
    vocab::Vocabulary{Data}
    config::Dict
end


function Recombine(vocab::Vocabulary{T}, config; embeddings=nothing) where T<:DataSet
    aij_k = Embed(att_weights(input=2*config["Kpos"]+1, output=config["attdim"]))
    aij_v = Embed(param(config["H"],2*config["Kpos"]+1,; atype=arrtype))
    dinput =  config["E"]  + config["A"]
    Recombine{T}(load_embed(vocab, config, embeddings),
                LSTM(input=dinput, hidden=config["H"], dropout=config["pdrop"], numLayers=config["Nlayers"]),
                Linear(input=3config["H"], output=config["E"]),
                Multiply(input=config["E"], output=config["Z"]),
                LSTM(input=config["E"], hidden=config["H"],  bidirectional=false, numLayers=1),
                LSTM(input=config["E"], hidden=config["H"],  bidirectional=false, numLayers=1),
                Attention(memory=config["H"], query=config["H"], att=config["H"]),
        #        Attention(memory=config["H"], query=config["H"], att=config["H"]),
                Linear(input=2config["H"], output=2config["Z"]),
                Param(zeroarray(arrtype,config["H"],config["Nlayers"])),
                Param(zeroarray(arrtype,config["H"],config["Nlayers"])),
                PositionalAttention(memory=config["H"],query=config["H"]+config["A"], att=config["attdim"], aij_k=aij_k, aij_v=aij_v),
        #        PositionalAttention(memory=config["H"],query=config["H"], att=config["attdim"], aij_k=aij_k, aij_v=aij_v),
                Pw{Float32}(2config["Z"], config["Kappa"]),
                vocab,
                config)
end

calc_ppl(model::Recombine, data) = train!(model, data; eval=true, dev=nothing)
calc_mi(model::Recombine, data)  = nothing
calc_au(model::Recombine, data)  = nothing
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
                    # @show "here"
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
                tokens = setdiff(xtokens,ntokens)
                cw     = String(map(UInt8,tokens))
                cnt    = 0
                for l in inds
                    if l != n && l !=i
                        w    = trnwords[l]
                        diff = compare(cw,w,dist)
                        if diff > thresh # 0.5
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

function beam_decode(model::Recombine, x, xp, xpp, z, protos; forviz=false)
    T,B         = eltype(arrtype), size(z,2)
    H,V,E       = model.config["H"], length(model.vocab.tokens), model.config["E"]
    contexts    = (batch_last(xp.context), batch_last(xpp.context))
    masks       = (arrtype(xp.mask'*T(-1e18)), arrtype(xpp.mask'*T(-1e18)))
    input       = ones(Int,B,1) .* specialIndicies.bow #x[:,1:1]
    #attentions = (zeroarray(arrtype,H,B), zeroarray(arrtype,H,B))
    states      = (expand_hidden(model.h0,B), expand_hidden(model.c0,B))
    limit       = model.config["maxLength"]
    traces      = [(zeros(1, B), states, ones(Int, B, limit), input, nothing, nothing)]
    for i=1:limit
        traces = beam_search(model, traces, contexts, masks, z, protos; step=i, forviz=forviz)
    end
    outputs     = map(t->t[3], traces)
    probs       = map(t->vec(t[1]), traces)
    score_arr, output_arr= traces[1][5], traces[1][6]
    return outputs, probs, score_arr,  output_arr
end

function cat_copy_scores(m::Recombine, y, scores)
    if m.config["copy"]
        vcat(y,scores[1], scores[2])
    else
        y
    end
end

function realindex(V::Integer, xp_tokens::AbstractVector{<:Integer}, xpp_tokens::AbstractVector{<:Integer}, index::Integer)
    if index > V
        if index > V + length(xp_tokens)
            return xpp_tokens[index-V-length(xp_tokens)]
        else
            return xp_tokens[index-V]
        end
    else
        return index
    end
end

realindex(model::Recombine, xp::Matrix{Int}, xpp::Matrix{Int}, indices::Vector{Int}) =
    map(ki->realindex(length(model.vocab.tokens), view(xp,ki[1],:), view(xpp,ki[1],:), ki[2]), enumerate(indices))

function sumprobs(output, xp::AbstractMatrix{Int}, xpp::AbstractMatrix{Int})
    L, B = size(output)
    Tp, Tpp = size(xp,2), size(xpp,2)
    V = L-Tp-Tpp
    for i=1:B
        for t=1:Tp
            output[xp[i,t],i] += output[V+t,i]
        end
        for t=1:Tpp
            output[xpp[i,t],i] += output[V+Tp+t,i]
        end
    end
    y = output[1:V,:]
    return log.(y)
end

clip_index(x; K=16) = max(-K, min(K, x)) + K + 1

function position_calc(L, step; K=16)
    clip_index.(collect(1:L) .- step; K=K)
end

function beam_search(model::Recombine, traces, contexts, masks, z, protos; step=1, forviz=false )
    result_traces = []
    bw = model.config["beam_width"]
    Kpos = model.config["Kpos"]
    for i=1:length(traces)
        probs, states, preds, cinput, scores_arr, output_arr = traces[i]
        if model.config["positional"]
            positions = map(p->position_calc(size(p,2), step; K=Kpos), protos)
        else
            positions = map(x->nothing, protos)
        end
        y, _,scores,states,weights  = decode_onestep(model, states, contexts, masks, z, cinput, positions)
        if step == 1
            y[1:6,:]  .= -1.0f18
        else
            y[1:2,:]  .= -1.0f18; y[4,:] .= -1.0f18;
        end
        yp                                 = cat_copy_scores(model, y, scores)
        out_soft                           = convert(Array, softmax(yp,dims=1))
        output                             = sumprobs(copy(out_soft), protos[1], protos[2])
        srtinds                            = mapslices(x->sortperm(x; rev=true), output,dims=1)[1:bw,:]
        cprobs                             = sort(output; rev=true, dims=1)[1:bw,:]
        inputs                             = srtinds
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
        push!(result_traces, ((cprobs .+ probs), states, preds, inputs, scores_softmaxed, out_soft, scores_arr, output_arr))
    end
    global_probs     = vcat(map(first, result_traces)...)
    global_srt_inds  = mapslices(x->sortperm(x; rev=true), global_probs, dims=1)[1:bw,:]
    global_srt_probs = sort(global_probs; rev=true, dims=1)[1:bw,:]
    new_traces = []
    for i=1:bw
        probs      = global_srt_probs[i:i,:]
        inds       = map(s->divrem(s,bw),global_srt_inds[i,:] .- 1)

        states     = [cat((result_traces[trace+1][2][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds))..., dims=2) for k=1:2]
        inputs     = vcat((result_traces[trace+1][4][loc+1,bi] for (bi, (trace, loc)) in enumerate(inds))...)
        if step == 1
            old_preds  = copy(result_traces[1][3])
        else
            old_preds  = vcat((result_traces[trace+1][3][bi:bi,:] for (bi, (trace, _)) in enumerate(inds))...)
        end
        #old_preds[:,step] .= realindex(model, protos[1], protos[2], inputs)
        old_preds[:,step] .= inputs

        if forviz
            scores  = [hcat((result_traces[trace+1][5][k][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...) for k=1:2]
            outsoft = hcat((result_traces[trace+1][6][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...)
            outsoft = reshape(outsoft, size(outsoft)...,1)
            scores  = map(s->reshape(s,size(s)...,1),scores)

            if step == 1
                scores_arr    = scores
                output_arr    = outsoft
            else
                old_scores    = [cat([result_traces[trace+1][7][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2) for k=1:2]
                old_outputs   = cat([result_traces[trace+1][8][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2)
                scores_arr    = [cat(old_scores[i],scores[i],dims=3) for i=1:2]
                output_arr    = cat(old_outputs,outsoft,dims=3)
            end
        else
            scores_arr, output_arr = nothing, nothing
        end
        push!(new_traces, (probs,states,old_preds,old_preds[:,step:step], scores_arr, output_arr))
    end
    return new_traces
end


function decode(model::Recombine, x, xp, xpp, z, protos; sampler=sample, training=true)
    T,B        = eltype(arrtype), size(z,2)
    H,V,E      = model.config["H"], length(model.vocab.tokens), model.config["E"]
    Kpos       = model.config["Kpos"]
    contexts   = (batch_last(xp.context), batch_last(xpp.context))
    masks      = (arrtype(xp.mask'*T(-1e18)),  arrtype(xpp.mask'*T(-1e18)))
    input      = x[:,1:1] #BOW
    #attentions = (zeroarray(arrtype,H,B), zeroarray(arrtype,H,B))
    states     = (expand_hidden(model.h0,B), expand_hidden(model.c0,B))
    limit      = (training ? size(x,2)-1 : model.config["maxLength"])
    preds      = ones(Int, B, limit)
    outputs    = []
    for i=1:limit
         if model.config["positional"]
             positions = map(p->position_calc(size(p,2), i; K=Kpos), protos)
         else
             positions = map(x->nothing, protos)
         end
         y, _, scores, states = decode_onestep(model, states, contexts, masks, z, input, positions)
         if !training
             if i == 1
                 y[1:6,:]  .= -1.0f18
             else
                 y[1:2,:]  .= -1.0f18; y[4,:] .= -1.0f18
             end
         end
         output   = cat_copy_scores(model, y, scores)
         push!(outputs, output)
         if !training
             output        = convert(Array, softmax(output,dims=1))
             output        = exp.(sumprobs(output, protos[1], protos[2]))
             preds[:,i]    = vec(mapslices(sampler, output, dims=1))
             input         = preds[:,i:i]
         else
             input         = x[:,i+1:i+1]
         end
    end
    Tp = sum(s->size(s,2),protos)
    return reshape(cat1d(outputs...),V+Tp,B,limit), preds
end

function decode_train(model::Recombine, x, xp, xpp, z, protos)
    T,B                 = eltype(arrtype), size(z,2)
    H,V,E               = model.config["H"], length(model.vocab.tokens), model.config["E"]
    Kpos                = model.config["Kpos"]
    pdrop, attdrop, outdrop = model.config["pdrop"], model.config["attdrop"], model.config["outdrop"]
    contexts            = (xp.context,xpp.context)
    masks               = (arrtype(xp.mask'*T(-1e18)),  arrtype(xpp.mask'*T(-1e18)))
    limit               = size(x,2)-1
    preds               = ones(Int, B, limit)
    z3d                 = zeroarray(arrtype, 1, 1, limit) .+ reshape(z, (size(z)...,1))
    xinput              = vcat(model.embed(x[:,1:end-1]), z3d)
    out                 = model.decoder(xinput, expand_hidden(model.h0,B), expand_hidden(model.c0,B))
    htop                = vcat(out.y,z3d)
    if model.config["positional"]
        positions = map(p->hcat((position_calc(size(p,2), i; K=Kpos) for i=1:limit)...), protos) # T x T'
    else
        positions = map(x->nothing, protos)
    end
    xp_attn, xp_score, _   = newattention(model.xp_attention, contexts[1],htop;mask=masks[1], pdrop=attdrop, positions=positions[1]) #positions[1]
    xpp_attn, xpp_score, _ = newattention(model.xp_attention, contexts[2],htop;mask=masks[2], pdrop=attdrop, positions=positions[2])
    if model.config["outdrop_test"]
        y               = model.output(vcat(dropout(out.y, outdrop; drop=true),xp_attn,xpp_attn))
    else
        y               = model.output(vcat(dropout(out.y, outdrop),xp_attn,xpp_attn))
    end
    yv                  = reshape(At_mul_B(model.embed.weight,mat(y,dims=1)),V,B,limit)
    output              = cat_copy_scores(model, yv, (xp_score,xpp_score))
    return output, preds
end

function decode_onestep(model::Recombine, states, contexts, masks, z, input, positions)
    pdrop, attdrop,outdrop   = model.config["pdrop"], model.config["attdrop"], model.config["outdrop"]
    e                         = mat(model.embed(input),dims=1)
    xi                        = vcat(e, z)
    out                       = model.decoder(xi, states[1], states[2])
    h, c                      = out.hidden, out.memory
    hbottom                   = vcat(out.y,z)
    xp_attn, xp_score, xp_weight         = model.xp_attention(contexts[1],hbottom;mask=masks[1], pdrop=attdrop, positions=positions[1]) #positions[1]
    xpp_attn, xpp_score, xpp_weight       = model.xp_attention(contexts[2],hbottom;mask=masks[2], pdrop=attdrop, positions=positions[2])
    if model.config["outdrop_test"]
        y                     = model.output(vcat(dropout(out.y, outdrop; drop=true),xp_attn,xpp_attn))
    else
        y                     = model.output(vcat(dropout(out.y, outdrop),xp_attn,xpp_attn))
    end
    yv                        = At_mul_B(model.embed.weight,y)
    yv, (xp_attn, xpp_attn), (xp_score, xpp_score), (h,c), (xp_weight, xpp_weight)
end

function encode(m::Recombine, x, xp, xpp; prior=false)
    pdrop =  m.config["pdrop"]
    xp_context    = m.xp_encoder(dropout(m.embed(xp.tokens), pdrop)).y
    xpp_context   = m.xp_encoder(dropout(m.embed(xpp.tokens),pdrop)).y
    if !m.config["kill_edit"]
        if prior
            μ     = zeroarray(arrtype,2latentsize(m),size(x.tokens,1))
        else
            x_context     = m.x_encoder(m.embed(x.tokens)).y
            x_embed       = source_hidden(x_context,x.lengths)
            xp_embed      = source_hidden(xp_context,xp.lengths)
            xpp_embed     = source_hidden(xpp_context,xpp.lengths)
            μ             = m.z_emb(vcat(m.x_xp_inter(xp_embed,x_embed)[1],
                                         m.x_xp_inter(xpp_embed,x_embed)[1]))
        end
        z         = sample_vMF(μ, m.config["eps"], m.config["max_norm"], m.pw; prior=prior)
    else
        z         = zeroarray(arrtype,2latentsize(m),xp.batchSizes[1])
    end
    return z, xp_context, xpp_context
end

function ind2BT(write_inds, copy_inds, H, BT)
    write_batches  = map(ind->(ind ÷ H) + 1, write_inds .- 1)
    copy_batches   = map(ind->(ind ÷ H) + 1, copy_inds .- 1)
    inds = [[write_inds[findall(b->b==i, write_batches)]; copy_inds[findall(b->b==i, copy_batches)]] for i=1:BT]
    inds_masked = PadSequenceArray(inds, pad=specialIndicies.mask)[1]
    inds_mask   = get_mask_sequence(length.(inds); makefalse=false)
    return (tokens=inds_masked, mask=inds_mask, sumind=findall(length.(inds) .> 0))
end

function getvalues(arr, values)
    if isempty(values)
        return 0f0
    else
        return sum(arr[values])
    end
end

function loss(model::Recombine, data; average=false, training=true)
    x, xp, xpp, copyinds, unbatched = data
    B = length(first(unbatched))
    copy_indices  = cat1d(copyinds...)
    z, xp_context, xpp_context = encode(model, x, xp, xpp; prior=false)
    Txp  = (mask=xp.mask, context=xp_context)
    Txpp = (mask=xpp.mask, context=xpp_context)
    y, _ = decode_train(model, x.tokens, Txp, Txpp, z, (xp.tokens, xpp.tokens))
    xmask = x.mask[:, 2:end]
    if training
        xmask = xmask .* (rand(size(xmask)...) .> model.config["rwritedrop"])
    end
    write_indices = findindices(y, (x.tokens[:, 2:end] .* xmask), dims=1)
    probs = softmax(mat(y, dims=1), dims=1) # H X BT
    # write_probs = probs[write_indices]
    # copy_probs  = probs[copy_indices]
    inds = ind2BT(write_indices, copy_indices, size(probs)...)
    marginals = log.(sum(probs[inds.tokens] .* arrtype(inds.mask),dims=2) .+ 1e-12)
    -sum(marginals[inds.sumind]) / B
    #-sum((log(write_probs[first(w)] + getvalues(copy_probs,c)) for (w,c) in ind2bt if !isempty(w))) / B
end


# function loss(model::Recombine, data; average=false)
#     x, xp, xpp, copyinds, unbatched = data
#     B = length(first(unbatched))
#     copy_indices  = cat1d(copyinds...)
#     z, xp_context, xpp_context = encode(model, x, xp, xpp; prior=false)
#     Txp  = (mask=xp.mask, context=xp_context)
#     Txpp = (mask=xpp.mask, context=xpp_context)
#     y, _ = decode_train(model, x.tokens, Txp, Txpp, z, (xp.tokens, xpp.tokens))
#     write_indices = findindices(y, (x.tokens[:, 2:end] .* x.mask[:, 2:end]), dims=1)
#     nlly = logp(mat(y, dims=1), dims=1)
#     -sum(nlly[[write_indices;copy_indices]])/B
# end

function copy_indices(xp, xmasked, L::Integer, offset::Integer=0)
    indices = Int[]
    B,T = size(xmasked)
    for t=2:T
        for i=1:B
            start = (t-2) * L * B + (i-1)*L + offset
            token =  xmasked[i,t]
            if token == specialIndicies.eow || token ∉ specialIndicies
                for i in findall(t->t==token,xp[i])
                    push!(indices, start+i)
                end
            end
        end
    end
    return indices
end

function trimencoded(x)
    stop = findfirst(i->i==specialIndicies.eow,x)
    stop = isnothing(stop) ? length(x) : stop
    return x[1:stop-1]
end


function sample(model::Recombine, data; N=nothing, sampler=sample, prior=true, sanitize=false, beam=false, forviz=false)
    N  = isnothing(N) ? model.config["N"] : N
    B  = max(model.config["B"],128)
    dt = Iterators.Stateful(Iterators.cycle(shuffle(data)))
    vocab = model.vocab
    samples = []
    for i in  progress(1 : (N ÷ B) + 1)
        b =  min(N,B)
        if (d = getbatch_recombine(model, dt,b)) !== nothing
            x, xp, xpp, copymasks, unbatched = d
            B = length(first(unbatched))
            z, xp_context, xpp_context = encode(model, x, xp, xpp; prior=prior)
            Txp  = (mask=xp.mask, context=xp_context)
            Txpp = (mask=xpp.mask, context=xpp_context)
            if beam
                preds, probs, scores_arr, output_arr  = beam_decode(model, x.tokens, Txp, Txpp, z, (xp.tokens,xpp.tokens); forviz=forviz)
                if forviz
                    for i=1:b
                        pred = preds[1][i,:]
                        stop = findfirst(x->x==specialIndicies.eow,pred)
                        stop = isnothing(stop) ? length(pred) : stop
                        xtrimmed    = trimencoded(preds[1][i,:])
                        xps         = (trimencoded(xp.tokens[i,:]),trimencoded(xpp.tokens[i,:]))
                        attscores   = [score[1:length(xps[k]),i,1:length(xtrimmed)] for  (k,score) in enumerate(scores_arr)]
                        ixp_end     = length(model.vocab)+length(xps[1])
                        ixpp_start  = length(model.vocab)+length(xp.tokens[i,:])+1
                        ixpp_end    = length(model.vocab)+length(xp.tokens[i,:])+length(xps[2])
                        indices     = [collect(1:ixp_end); collect(ixpp_start:ixpp_end)]
                        outsoft     = output_arr[indices,i,1:length(xtrimmed)]
                        push!(samples, (x=xtrimmed, scores=attscores, probs=outsoft, xp=xps[1], xpp=xps[2]))
                    end
                    continue
                end
                probs2D = softmax(hcat(probs...), dims=2)
                s       = hcat(map(pred->vec(mapslices(x->trim(x,vocab), pred, dims=2)), preds)...)
                s2      = [trimencoded(preds[1][i,:]) for i=1:b]
                s       = mapslices(x->join(x,'\n'),s,dims=2)
                interesting = map(x->occursin("jump",x) || occursin("I_JUMP",x) , s)
            else
                y, preds   = decode(model, x.tokens, Txp, Txpp, z, (xp.tokens,xpp.tokens); sampler=sampler, training=false)
                s          = mapslices(x->trim(x,vocab), preds, dims=2)
                probs2D    = ones(length(s),1)
            end
            for i=1:b
            #    if interesting[i]
                    push!(samples, (target    = join(vocab.tokens[unbatched[1][i]],' '),
                                    xp        = join(vocab.tokens[unbatched[2][i]],' '),
                                    xpp       = join(vocab.tokens[unbatched[3][i]],' '),
                                    sample    = s[i],
                                    sampleenc = s2[i],
                                    probs     = probs2D[i,:] ))
            #    end
            end
        end
    end
    return samples
end


limit_seq_length(x; maxL=30) = map(s->(length(s)>maxL ? s[1:Int(maxL)] : s) , x)

function getbatch_recombine(model, iter, B)
    edata   = collect(Iterators.take(iter,B))
    unk, mask, eow, bow, sep = specialIndicies
    maxL = model.config["maxLength"]
    V       = length(model.vocab.tokens)
    if (b = length(edata)) != 0
        d           = (x, xp1, xp2) = unzip(edata)
        xp,xpp      = (xp1,xp2)  #rand()>0.5 ? (xp1,xp2) : (xp2,xp1)
        x           = limit_seq_length(x;maxL=maxL)
        xp          = map(s->[s;eow], limit_seq_length(xp;maxL=maxL))
        xpp         = map(s->[s;eow], limit_seq_length(xpp;maxL=maxL))
        pxp         = PadSequenceArray(xp, pad=mask, makefalse=true)
        pxpp        = PadSequenceArray(xpp, pad=mask, makefalse=true)
        px          = PadSequenceArray(map(xi->[bow;xi;eow], x), pad=mask, makefalse=false)
        seq_xp      = (tokens=pxp[1],
                       mask=pxp[2],
                       lengths=length.(xp))
        seq_xpp     = (tokens=pxpp[1],
                       mask=pxpp[2],
                       lengths=length.(xpp))
        seq_x       = (tokens=px[1],
                       mask=px[2],
                       lengths=length.(x))
        Tp, Tpp     = size(seq_xp.tokens,2),  size(seq_xpp.tokens,2)
        L = V + Tp + Tpp
        xp_copymask  = copy_indices(xp, seq_x.tokens,  L, V)
        xpp_copymask = copy_indices(xpp, seq_x.tokens, L, V+Tp)
        return seq_x, seq_xp, seq_xpp, (xp_copymask, xpp_copymask), d
    end
    return nothing
end



function train!(model::Recombine, data; eval=false, dev=nothing)
    bestparams = deepcopy(parameters(model))
    setoptim!(model,model.config["optim"])
    ppl = typemax(Float64)
    if !eval
        model.config["rpatiance"] = model.config["patiance"]
        model.config["rwritedrop"] = model.config["writedrop"]
    end
    total_iter = 0
    for i=1:(eval ? 1 : model.config["epoch"])
        lss, ntokens, ninstances = 0.0, 0.0, 0.0
        dt  = Iterators.Stateful(shuffle(data))
        msg(p) = string(@sprintf("Iter: %d,Lss(ptok): %.2f,Lss(pinst): %.2f, PPL(test): %.2f", total_iter, lss/ntokens, lss/ninstances, ppl))
        for i in progress(msg, 1:(length(dt) ÷ model.config["B"]))
            total_iter += 1
            d = getbatch_recombine(model, dt,model.config["B"])
            b = size(d[1].mask,1)
            n = sum(d[1].mask[:,2:end]) + sum(length,d[4])
            if !eval
                J = @diff loss(model, d; average=false)
                for w in parameters(J)
                    g = grad(J,w)
                    if !isnothing(g)
                        KnetLayers.update!(value(w), g, w.opt)
                    end
                end
                #model.config["rwritedrop"] = max(model.config["rwritedrop"] - 0.0001,model.config["writedrop"])
            else
                J = loss(model, d; average=false, training=false)
            end
            lss        += (value(J)*b)
            ntokens    += n
            ninstances += b
            if !eval && i%500==0
                #print_ex_samples(model, data; beam=true)
                if !isnothing(dev)
                     #print_ex_samples(model, dev; beam=true)
                     #calc_ppl(model, dev)
                end
            end
        end
        if !isnothing(dev)
            newppl = calc_ppl(model, dev)
            @show newppl
            if newppl > ppl
                lrdecay!(model, model.config["lrdecay"])
                model.config["rpatiance"] = model.config["rpatiance"] - 1
                println("patiance decay, rpatiance: $(model.config["rpatiance"])")
                if model.config["rpatiance"] == 0
                    break
                end
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
            ppl = exp(lss/ntokens + kl_calc(model)*ninstances/ntokens)
        end
#        total_iter > 400000 && break
    end
    if !isnothing(dev)
        for (best, current) in zip(bestparams,parameters(model))
            copyto!(value(current),value(best))
        end
    end
    return ppl
end



recombine_config = Dict(
               "model"=> Recombine,
               "lang"=>"turkish",
               "kill_edit"=>false,
               "attend_pr"=>0,
               "A"=>32,
               "H"=>256,
               "Z"=>16,
               "E"=>64,
               "B"=>32,
               "attdim"=>32,
               "Kpos" =>16,
               "concatz"=>true,
               "optim"=>Adam(lr=0.002, gclip=10.0),
               "kl_weight"=>0.0,
               "kl_rate"=> 0.05,
               "fb_rate"=>4,
               "N"=>10000,
               "useprior"=>true,
               "aepoch"=>1, #20
               "epoch"=>10,  #40
               "Ninter"=>10,
               "pdrop"=>0.5,
               "calctrainppl"=>false,
               "Nsamples"=>100,
               "pplnum"=>1000,
               "authresh"=>0.1,
               "Nlayers"=>2,
               "Kappa"=>25,
               "max_norm"=>10.0,
               "eps"=>1.0,
               "activation"=>ELU,
               "maxLength"=>45,
               "calc_trainppl"=>false,
               "num_examplers"=>2,
               "dist_thresh"=>0.5,
               "max_cnt_nb"=>5,
               "task"=>SCANDataSet,
               "patiance"=>4,
               "lrdecay"=>0.5,
               "conditional" => true,
               "split" => "add_prim",
               "splitmodifier" => "jump",
               "beam_width" => 4,
               "copy" => true,
               "writedrop" => 0.5,
               "outdrop" => 0.7,
               "attdrop" => 0.1,
               "outdrop_test" => true,
               "positional" => true,
               "masktags" => false
               )

function vmfKL(m::Recombine)
   k, d = m.config["Kappa"], 2m.config["Z"]
   k*((besseli(d/2.0+1.0,k) + besseli(d/2.0,k)*d/(2.0*k))/besseli(d/2.0, k) - d/(2.0*k)) +
   d * log(k)/2.0 - log(besseli(d/2.0,k)) -
   loggamma(d/2+1) - d * log(2)/2
end

function pickprotos(model, eset)
    set    = map(d->xfield(model.config["task"],d,model.config["conditional"]),eset)
    set    = unique(set)
    inputs = Set(map(d->join(model.vocab.tokens[d[1:findfirst(s->s==specialIndicies.sep,d)-1]],' '), set))
    outputs = Set(map(d->join(model.vocab.tokens[d[findfirst(s->s==specialIndicies.sep,d)+1:end]],' '), set))
    data = []
    for i=1:length(set)
        for j=1:length(set)
            j==i && continue
            if length(set[j]) + length(set[i]) < 29 # average training length
                push!(data, (x=Int[specialIndicies.bow], xp=set[i], xpp=set[j]))
            end
        end
    end
    data, inputs, outputs
end

function print_samples(model, eset; beam=true, fname="samples.txt", N=400)
    data, train_inputs, train_outputs = pickprotos(model, eset)
    inpdict, outdict = model.vocab.inpdict.toIndex, model.vocab.outdict.toIndex
    samples = sample(model, data; N=N*300, sampler=argmax, prior=true, beam=beam)
    cnt = 0
    f = open(fname, "w+")
    for s in samples
         if s.probs[1] > (1.2 / model.config["beam_width"])
            datum = split(s.sample,'\n')[1]
            if occursin("→", datum) && length(findall(d->d=='→', datum)) == 1
                line = "IN: "*replace(datum,"→"=> "OUT:")
                endswith(line,":") && continue
                input, output = parseDataLine(line, model.vocab.parser)
                if join(input,' ') ∉ train_inputs && join(output,' ') ∉ train_outputs && all(i->haskey(inpdict,i), input) &&  all(i->haskey(outdict,i), output)
                    println(f,line)
                    cnt+=1
                    if cnt == N
                        break
                    end
                end
            end
         end
    end
    close(f)
end

function to_jacobs_format(model, fname)
    lines = readlines(fname)
    data = map(line->parseDataLine(line, model.vocab.parser), readlines(fname))
    json = map(d->JSON.lower((inp=d.input,out=d.output)), data)
    fout = fname[1:first(findfirst(".", fname))-1] * ".json"
    open(fout, "w+") do f
        JSON.print(f,json, 4)
    end
end

using Plots, JSON
pyplot()

function viz(model, data; N=5)
    samples = sample(model, data; N=N, sampler=sample, prior=false, beam=true, forviz=true)
    vocab = model.vocab.tokens
    json = map(d->JSON.lower((x=vocab[d.x], xp=vocab[d.xp], xpp=vocab[d.xpp], scores1=d.scores[1], scores2=d.scores[2], probs=d.probs)), samples)
    open("attention_maps.json", "w+") do f
        JSON.print(f,json, 4)
    end
    for i=1:length(samples)
        x, scores, probs, xp, xpp = samples[i]
        attension_visualize(model.vocab, probs, scores, xp, xpp, x; prefix="$i")
        println("----------------------")
        println("x: ", join(model.vocab.tokens[x],' '))
        println("xp: ", join(model.vocab.tokens[xp],' '))
        println("xpp: ", join(model.vocab.tokens[xpp],' '))
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



function preprocess(m::Recombine{SIGDataSet}, train, devs...)
    println("preprocessing SIGDataSet")
    T = SIGDataSet
    dist = Levenshtein()
    thresh, cond, maxcnt, masktags =  m.config["dist_thresh"], m.config["conditional"], m.config["max_cnt_nb"], m.config["masktags"]
    sets = map((train,devs...)) do set
            map(d->(x=xfield(T,d,cond; masktags=masktags),lemma=d.lemma, surface=d.surface, tags=(masktags ?  fill!(similar(d.tags),specialIndicies.mask) : d.tags)),set)
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
                        tdiff       = compare(xtag,tag,dist)
                        trealdiff   = setdiff(difftag,tag_tokens)
                        if length(trealdiff) == 0
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


function print_samples_sig(model, eset;beam=true, fname="samples.txt", N=400)
    data, train_inputs, train_outputs = pickprotos(model, eset)
    inpdict, outdict = model.vocab.inpdict.toIndex, model.vocab.outdict.toIndex
    samples = sample(model, data; N=N*300, sampler=argmax, prior=true, beam=beam)
    tags = Set(setdiff(model.vocab.inpdict.toElement,model.vocab.outdict.toElement))
    cnt = 0
    open(fname, "w+") do f
        for s in samples
             if s.probs[1] > (1.2 / model.config["beam_width"])
                datum = model.vocab.tokens[s.sampleenc]
                if "→" ∈ datum && length(findall(d->d=="→", datum)) == 1
                    tokens = datum
                    index = findfirst(t->t=="→",tokens)
                    input,output = tokens[1:index-1], tokens[index+1:end]
                    indexsep = findfirst(t->t ∈ tags,input)
                    lemma, ts = input[1:indexsep-1], input[indexsep+1:end]
                    (length(ts) == 0 || length(lemma) == 0 || length(output) == 0) && continue
                    line = "$(join(lemma))\t$(join(output))\t$(join(ts,';'))"
                    input, output = parseDataLine(line, model.vocab.parser)
                    if join(input,' ') ∉ train_inputs && join(output,' ') ∉ train_outputs && all(i->haskey(inpdict,i), input) &&  all(i->haskey(outdict,i), output)
                        println(f,line)
                        cnt+=1
                        if cnt == N
                            break
                        end
                    end
                end
             end
        end
    end
end

function set_plot_font(fpath="Symbola.ttf")
    font_manager = Plots.PyPlot.matplotlib["font_manager"]
    font_dirs = ["$(pwd())"]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    font_list = font_manager.createFontList(font_files)
    @show font_list
    push!(font_manager.fontManager.ttflist, font_list...)
        Plots.PyCall.PyDict(Plots.PyPlot.matplotlib["rcParams"])["font.family"] ="sans-serif"
    Plots.PyCall.PyDict(Plots.PyPlot.matplotlib["rcParams"])["font.sans-serif"] =["Symbola"]
end
