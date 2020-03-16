struct PositionalAttention
    key_transform::Dense
    query_transform::Dense
    memory_transform::Dense
    aij_k::Union{Embed,Nothing}
    aij_v::Union{Embed,Nothing}
    attdim::Int
end

function att_weight_init(output::Int, input::Int)
    st = (1 / sqrt(input))
    2st .* rand!(arrtype(undef,output,input)) .- st
end

function PositionalAttention(;memory::Int,query::Int,att::Int, aij_k=nothing, aij_v=nothing)
    PositionalAttention(Dense(input=memory,output=att, winit=att_weight_init, activation=Tanh()),
              Dense(input=query,output=att, winit=att_weight_init, activation=Tanh()),
              Dense(input=memory,output=memory, winit=att_weight_init, activation=Tanh()),
              aij_k,
              aij_v,
              att)
end

function ewise_interact(m::PositionalAttention, memory, query; pdrop=0.0)
    tkey     = m.key_transform(memory)
    tquery   = m.query_transform(query)
    score    = dropout(tquery .* tkey, pdrop) ./ sqrt(m.attdim) #[H,B]
    weights  = softmax(score;dims=1) #[H,B]
    values   = m.memory_transform(memory) .* weights # [H,B]
    return (values, weights)
end

function (m::PositionalAttention)(memory, query; pdrop=0, mask=nothing, positions=nothing)
    tquery  = m.query_transform(query) # [A,B,T] or [A,B]
    memory  = batch_last(memory)
    tkey    = m.key_transform(memory)
    pscores = nothing
    if ndims(query) == 2
        if !isnothing(positions)
            pscores   = expand(At_mul_B(m.aij_k(positions), tquery), dim=3)
        end
        tquery = expand(tquery, dim=2)
    else
        if !isnothing(positions)
            pscores   = bmm(m.aij_k(positions), tquery; transA=true)
        end
        tquery = batch_last(tquery)
    end

    scores = batch_last(bmm(tkey, tquery,transA=true)) #[T',B,T]

    if !isnothing(positions)
        scores += pscores
    end

    scores = dropout(scores, pdrop) ./ sqrt(m.attdim)

    if !isnothing(mask)
        scores = applymask2(scores, expand(mask, dim=3), +) # size(mask) == [T', B , 1]
    end

    #memory H,B,T' # weights T' X B X T
    weights  = softmax(scores;dims=1) #[T',B,T]
    values   = batch_last(bmm(m.memory_transform(memory),batch_last(weights))) # [H,T,B]
    if !isnothing(positions)
        pmemory = m.aij_v(positions)
        if ndims(query) == 2
            pvalues = expand(pmemory * mat(weights; dims=1), dim=3)
        else
            pvalues = bmm(pmemory, weights) # H X B X T times
        end
        values += pvalues
    end

    if  ndims(query) == 2
        mat(values, dims=1), mat(scores,dims=1), mat(weights, dims=1)
    else
        values, scores, weights
    end
end

struct Recombine{Data}
    embed::Embed
    decoder::LSTM
    output::Linear
    enclinear::Multiply
    xp_encoder::LSTM
    x_encoder::LSTM
    x_xp_inter::PositionalAttention
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
                PositionalAttention(memory=config["H"], query=config["H"], att=config["H"]),
        #        Attention(memory=config["H"], query=config["H"], att=config["H"]),
                Linear(input=2config["H"], output=2config["Z"]),
                Param(zeroarray(arrtype,config["H"],config["Nlayers"])),
                Param(zeroarray(arrtype,config["H"],config["Nlayers"])),
                PositionalAttention(memory=config["H"],query=config["H"]+config["A"]+config["E"], att=config["attdim"], aij_k=aij_k, aij_v=aij_v),
        #        PositionalAttention(memory=config["H"],query=config["H"], att=config["attdim"], aij_k=aij_k, aij_v=aij_v),
                Pw{Float32}(2config["Z"], config["Kappa"]),
                vocab,
                config)
end

calc_ppl(model::Recombine, data)    = train!(model, data; eval=true, dev=nothing)
calc_mi(model::Recombine, data)     = nothing
calc_au(model::Recombine, data)     = nothing
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

function beam_decode(model::Recombine, x, xp, xpp, z; forviz=false)
    T,B         = eltype(arrtype), size(z,2)
    H,V,E       = model.config["H"], length(model.vocab.tokens), model.config["E"]
    #contexts    = (xp.context,xpp.context)
    masks       = (arrtype(xp.mask'*T(-1e18)), arrtype(xpp.mask'*T(-1e18)))
    input       = ones(Int,B,1) .* specialIndicies.bow #x[:,1:1]
    #attentions = (zeroarray(arrtype,H,B), zeroarray(arrtype,H,B))
    states      = (_repeat(model.h0,B,dim=2), _repeat(model.c0,B,dim=2))
    limit       = model.config["maxLength"]
    traces      = [(zeros(1, B), states, ones(Int, B, limit), input, nothing, nothing)]
    protos      = (xp=xp,xpp=xpp)
    for i=1:limit
        traces = beam_search(model, traces, protos, masks, z; step=i, forviz=forviz)
    end
    outputs     = map(t->t[3], traces)
    probs       = map(t->vec(t[1]), traces)
    score_arr, output_arr = traces[1][5], traces[1][6]
    return outputs, probs, score_arr,  output_arr
end


cat_copy_scores(m::Recombine, y, scores) =
    m.config["copy"] ? vcat(y,scores[1], scores[2]) : y

function sumprobs(output, xp::AbstractMatrix{Int}, xpp::AbstractMatrix{Int})
    L, B = size(output)
    Tp, Tpp = size(xp,2), size(xpp,2)
    V = L-Tp-Tpp
    for i=1:B
        for t=1:Tp
            @inbounds output[xp[i,t],i]  += output[V+t,i]
        end
        for t=1:Tpp
            @inbounds output[xpp[i,t],i] += output[V+Tp+t,i]
        end
    end
    return log.(output[1:V,:])
end

clip_index(x; K=16) = max(-K, min(K, x)) + K + 1
position_calc(L, step; K=16) = clip_index.(collect(1:L) .- step; K=K)
function beam_search(model::Recombine, traces, protos, masks, z; step=1, forviz=false )
    result_traces = []
    bw = model.config["beam_width"]
    Kpos = model.config["Kpos"]
    positions = (nothing,nothing)
    @inbounds for i=1:length(traces)
        probs, states, preds, cinput, scores_arr, output_arr = traces[i]
        if model.config["positional"]
            positions = map(p->position_calc(size(p.tokens,2), step; K=Kpos), protos)
        end
        y,_,scores,states,weights  = decode_onestep(model, states, protos, masks, z, cinput, positions)
        step == 1 ? negativemask!(y,1:6) : negativemask!(y,1:2,4)
        yp  = cat_copy_scores(model, y, scores)
        out_soft  = convert(Array, softmax(yp,dims=1))
        output = sumprobs(copy(out_soft), protos[1].tokens, protos[2].tokens)
        srtinds = mapslices(x->sortperm(x; rev=true), output, dims=1)[1:bw,:]
        cprobs = sort(output; rev=true, dims=1)[1:bw,:]
        inputs = srtinds
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
    global_probs     = mapreduce(first, vcat, result_traces)
    global_srt_inds  = mapslices(x->sortperm(x; rev=true), global_probs, dims=1)[1:bw,:]
    global_srt_probs = sort(global_probs; rev=true, dims=1)[1:bw,:]
    new_traces = []
    @inbounds for i=1:bw
        probs      = global_srt_probs[i:i,:]
        inds       = map(s->divrem(s,bw), global_srt_inds[i,:] .- 1)

        states     = ntuple(k->cat((result_traces[trace+1][2][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds))..., dims=2),2)
        inputs     = vcat((result_traces[trace+1][4][loc+1,bi] for (bi, (trace, loc)) in enumerate(inds))...)
        if step == 1
            old_preds  = copy(result_traces[1][3])
        else
            old_preds  = vcat((result_traces[trace+1][3][bi:bi,:] for (bi, (trace, _)) in enumerate(inds))...)
        end
        old_preds[:,step] .= inputs

        if forviz
            scores  = ntuple(k->hcat((result_traces[trace+1][5][k][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...),2)
            outsoft = hcat((result_traces[trace+1][6][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...)
            outsoft = reshape(outsoft,size(outsoft)...,1)
            scores  = map(s->reshape(s,size(s)...,1),scores)

            if step == 1
                scores_arr    = scores
                output_arr    = outsoft
            else
                old_scores    = ntuple(k->cat([result_traces[trace+1][7][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2),2)
                old_outputs   = cat([result_traces[trace+1][8][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2)
                scores_arr    = ntuple(i->cat(old_scores[i],scores[i],dims=3),2)
                output_arr    = cat(old_outputs,outsoft,dims=3)
            end
        else
            scores_arr, output_arr = nothing, nothing
        end
        push!(new_traces, (probs,states,old_preds,old_preds[:,step:step], scores_arr, output_arr))
    end
    return new_traces
end

function negativemask!(y,inds...)
    T = eltype(y)
    @inbounds for i in inds
        y[i,:] .= -T(1.0f18)
    end
end

function decode(model::Recombine, x, xp, xpp, z; sampler=sample, training=true)
    T,B        = eltype(arrtype), size(z,2)
    H,V,E      = model.config["H"], length(model.vocab.tokens), model.config["E"]
    Kpos       = model.config["Kpos"]
    protos     = (xp=xp,xpp=xpp)
    masks      = (arrtype(xp.mask'*T(-1e18)),  arrtype(xpp.mask'*T(-1e18)))
    input      = ones(Int,B,1) .* specialIndicies.bow
    states     = (_repeat(model.h0,B,dim=2), _repeat(model.c0,B,dim=2))
    limit      = (training ? size(x,2)-1 : model.config["maxLength"])
    preds      = ones(Int, B, limit)
    outputs    = []
    positions  = (nothing,nothing)
    for i=1:limit
         if model.config["positional"]
             positions = map(p->position_calc(size(p.tokens,2), i; K=Kpos), protos)
         end
         y, _,scores, states = decode_onestep(model, states, protos, masks, z, input, positions)
         if !training
             i == 1 ? negativemask!(y,1:6) : negativemask!(y,1:2,4)
         end

         output   = cat_copy_scores(model, y, scores)
         push!(outputs, output)

         if !training
             output        = convert(Array, softmax(output,dims=1))
             output        = sumprobs(output, protos[1].tokens, protos[2].tokens)
             preds[:,i]    = vec(mapslices(sampler, output, dims=1))
             input         = preds[:,i:i]
         else
             input         = x[:,i+1:i+1]
         end
    end
    Tp = sum(s->size(s.tokens,2),protos)
    return reshape(cat1d(outputs...),V+Tp,B,limit), preds
end

function decode_train(model::Recombine, x, xp, xpp, z)
    T,B                 = eltype(arrtype), size(z,2)
    H,V,E               = model.config["H"], length(model.vocab.tokens), model.config["E"]
    Kpos                = model.config["Kpos"]
    pdrop, attdrop, outdrop = model.config["pdrop"], model.config["attdrop"], model.config["outdrop"]
    masks               = (arrtype(xp.mask'*T(-1e18)),  arrtype(xpp.mask'*T(-1e18)))
    limit               = size(x,2)-1
    preds               = ones(Int, B, limit)
    z3d                 = zeroarray(arrtype, 1, 1, limit) .+ reshape(z, (size(z)...,1))
    xinput              = vcat(model.embed(x[:,1:end-1]), z3d)
    out                 = model.decoder(xinput, _repeat(model.h0,B,dim=2), _repeat(model.c0,B,dim=2))
    htop                = vcat(out.y,xinput)
    if model.config["positional"]
        positions = map(p->hcat((position_calc(size(p.tokens,2), i; K=Kpos) for i=1:limit)...),(xp,xpp)) # T x T'
    else
        positions = (nothing,nothing)
    end
    xp_attn, xp_score, _   = model.xp_attention(xp.context,htop;mask=masks[1], pdrop=attdrop, positions=positions[1]) #positions[1]
    xpp_attn, xpp_score, _ = model.xp_attention(xpp.context,htop;mask=masks[2], pdrop=attdrop, positions=positions[2])
    if model.config["outdrop_test"]
        y               = model.output(vcat(dropout(out.y, outdrop; drop=true),xp_attn,xpp_attn))
    else
        y               = model.output(vcat(dropout(out.y, outdrop),xp_attn,xpp_attn))
    end
    yv                  = reshape(At_mul_B(model.embed.weight,mat(y,dims=1)),V,B,limit)
    output              = cat_copy_scores(model, yv, (xp_score,xpp_score))
    return output, preds
end

function decode_onestep(model::Recombine, states, protos, masks, z, input, positions)
    pdrop, attdrop, outdrop = model.config["pdrop"], model.config["attdrop"], model.config["outdrop"]
    e  = mat(model.embed(input),dims=1)
    xi = vcat(e, z)
    out = model.decoder(xi, states...)
    hbottom = vcat(out.y,e,z)
    xp_attn, xp_score, xp_weight = model.xp_attention(protos.xp.context,hbottom;mask=masks[1], pdrop=attdrop, positions=positions[1]) #positions[1]
    xpp_attn, xpp_score, xpp_weight = model.xp_attention(protos.xpp.context,hbottom;mask=masks[2], pdrop=attdrop, positions=positions[2])
    ydrop = model.config["outdrop_test"] ? dropout(out.y, outdrop; drop=true) : dropout(out.y, outdrop)
    yc = model.output(vcat(ydrop,xp_attn,xpp_attn))
    yfinal = At_mul_B(model.embed.weight,yc)
    yfinal, (xp_attn, xpp_attn), (xp_score, xpp_score), (out.hidden, out.memory), (xp_weight, xpp_weight)
end

function encode(m::Recombine, x, xp, xpp; prior=false)
    pdrop =  m.config["pdrop"]
    xp_context    = m.xp_encoder(dropout(m.embed(xp.tokens), pdrop)).y
    xpp_context   = m.xp_encoder(dropout(m.embed(xpp.tokens),pdrop)).y
    if !m.config["kill_edit"]
        if prior
            μ     = zeroarray(arrtype,2latentsize(m),size(x.tokens,1))
        else
            x_context     = m.x_encoder(m.embed(x.tokens[:,2:end])).y
            x_embed       = source_hidden(x_context,x.lens)
            xp_embed      = source_hidden(xp_context,xp.lens)
            xpp_embed     = source_hidden(xpp_context,xpp.lens)
            μ             = m.z_emb(vcat(ewise_interact(m.x_xp_inter,xp_embed,x_embed)[1],
                                         ewise_interact(m.x_xp_inter,xpp_embed,x_embed)[1]))
        end
        z         = sample_vMF(μ, m.config["eps"], m.config["max_norm"], m.pw; prior=prior)
    else
        z         = zeroarray(arrtype,2latentsize(m),xp.batchSizes[1])
    end
    return z, (xp..., context=xp_context), (xpp..., context=xpp_context)
end

function ind2BT(write_inds, copy_inds, H, BT)
    write_batches  = map(ind->(ind ÷ H) + 1, write_inds .- 1)
    copy_batches   = map(ind->(ind ÷ H) + 1, copy_inds .- 1)
    inds = [[write_inds[findall(b->b==i, write_batches)]; copy_inds[findall(b->b==i, copy_batches)]] for i=1:BT]
    inds_masked, inds_mask  = PadSequenceArray(inds, pad=specialIndicies.mask, makefalse=false)
    return (tokens=inds_masked, mask=inds_mask, sumind=findall(length.(inds) .> 0))
end

function loss(model::Recombine, data; average=false, eval=false)
    x, xp, xpp, copyinds, unbatched = data
    B = length(first(unbatched))
    copy_indices  = cat1d(copyinds...)
    z, Txp, Txpp = encode(model, x, xp, xpp; prior=false)
    output, _ = decode_train(model, x.tokens, Txp, Txpp, z)
    ytokens, ymask = x.tokens[:,2:end], x.mask[:, 2:end]
    if !eval
        ymask = ymask .* (rand(size(ymask)...) .> model.config["rwritedrop"])
    end
    write_indices = findindices(output, (ytokens .* ymask), dims=1)
    probs = softmax(mat(output, dims=1), dims=1) # H X BT
    inds = ind2BT(write_indices, copy_indices, size(probs)...)
    marginals = log.(sum(probs[inds.tokens] .* arrtype(inds.mask),dims=2) .+ 1e-12)
    loss = -sum(marginals[inds.sumind]) / B
end

function copy_indices(xp, xmasked, L::Integer, offset::Integer=0)
    indices = Int[]
    B,T = size(xmasked)
    for t=2:T
        @inbounds for i=1:B
            start = (t-2) * L * B + (i-1)*L + offset
            token =  xmasked[i,t]
            if token == specialIndicies.eow || token ∉ specialIndicies
                @inbounds for i in findall(t->t==token,xp[i])
                    push!(indices, start+i)
                end
            end
        end
    end
    return indices
end


limit_seq_length(x; maxL=30) = map(s->(length(s)>maxL ? s[1:Int(maxL)] : s) , x)

function getbatch_recombine(model, iter, B)
    edata = collect(Iterators.take(iter,B))
    b = length(edata); b==0 && return nothing
    unk, mask, eow, bow, sep = specialIndicies
    maxL = model.config["maxLength"]
    V = length(model.vocab.tokens)
    d           = (x, xp1, xp2) = unzip(edata)
    xp,xpp      = (xp1, xp2) #rand()>0.5 ? (xp1,xp2) : (xp2,xp1)
    x           = map(s->[bow;s;eow],limit_seq_length(x;maxL=maxL))
    xp          = map(s->[s;eow], limit_seq_length(xp;maxL=maxL))
    xpp         = map(s->[s;eow], limit_seq_length(xpp;maxL=maxL))
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
                J = @diff loss(model, d)
                for w in parameters(J)
                    g = grad(J,w)
                    if !isnothing(g)
                        KnetLayers.update!(value(w), g, w.opt)
                    end
                end
                #model.config["rwritedrop"] = max(model.config["rwritedrop"] - 0.0001,model.config["writedrop"])
            else
                J = loss(model, d; eval=true)
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

            if newppl > ppl-0.0025 # not a good improvement
                lrdecay!(model, model.config["lrdecay"])
                model.config["rpatiance"] = model.config["rpatiance"] - 1
                println("patiance decay, rpatiance: $(model.config["rpatiance"])")
                if model.config["rpatiance"] == 0
                    break
                end
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





function vmfKL(m::Recombine)
   k, d = m.config["Kappa"], 2m.config["Z"]
   k*((besseli(d/2.0+1.0,k) + besseli(d/2.0,k)*d/(2.0*k))/besseli(d/2.0, k) - d/(2.0*k)) +
   d * log(k)/2.0 - log(besseli(d/2.0,k)) -
   loggamma(d/2+1) - d * log(2)/2
end



function to_json(model, fname)
    lines = readlines(fname)
    data = map(line->parseDataLine(line, model.vocab.parser), readlines(fname))
    json = map(d->JSON.lower((inp=d.input,out=d.output)), data)
    fout = fname[1:first(findfirst(".", fname))-1] * ".json"
    open(fout, "w+") do f
        JSON.print(f,json, 4)
    end
    return fout
end

using Plots
pyplot()

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
                        tdiff       = compare(trntags[n],tag,dist)
                        trealdiff   = setdiff(difftag,tag_tokens)
                        if length(trealdiff) == 0 &&  tdiff > 0.7
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

function std_mean(iter)
    μ   = mean(iter)
    σ2  = stdm(iter,μ)
    return  σ2, μ
end

function pickprotos(model::Recombine{SCANDataSet}, processed, esets)
    p_std, p_mean = std_mean((length(p.xp) for p in first(processed)))
    pthresh = p_mean-sqrt(p_std)
    pp_std, pp_mean = std_mean((length(p.xpp) for p in first(processed)))
    ppthresh = pp_mean+sqrt(pp_mean)
    eset    = length(esets) == 3  ? [esets[1];esets[3]] : esets[1]
    set     = map(d->xfield(model.config["task"],d,model.config["conditional"]),eset)
    set     = unique(set)
    inputs  = Set(map(d->join(model.vocab.tokens[d[1:findfirst(s->s==specialIndicies.sep,d)-1]],' '), set))
    outputs = Set(map(d->join(model.vocab.tokens[d[findfirst(s->s==specialIndicies.sep,d)+1:end]],' '), set))
    data = []
    for i=1:length(set)
        length(set[i]) < pthresh && continue
         for j=1:length(set)
            j==i && continue
            length(set[j]) > ppthresh && continue
            push!(data, (x=Int[specialIndicies.bow], xp=set[i], xpp=set[j]))
        end
    end
    data, inputs, outputs
end

function pickprotos(model::Recombine{SIGDataSet}, processed, esets)
    eset    = length(esets) == 3  ? [esets[1];esets[3]] : esets[1]
    set     = map(d->xfield(model.config["task"],d,model.config["conditional"]),eset)
    set     = unique(set)
    inputs  = map(d->join(model.vocab.tokens[d[1:findfirst(s->s==specialIndicies.sep,d)-1]],' '), set)
    outputs = map(d->join(model.vocab.tokens[d[findfirst(s->s==specialIndicies.sep,d)+1:end]],' '), set)
    data = []
    for i=1:length(set)
         p_inp = collect(inputs[i])
         indexsep = findfirst(t->t == specialTokens.iosep[1],p_inp)
         p_lemma, p_ts = p_inp[1:indexsep-1],p_inp[indexsep+1:end]
         for j=1:length(set)
            j==i && continue
            pp_inp = collect(inputs[j])
            indexsep = findfirst(t->t ∈ specialTokens.iosep[1],pp_inp)
            pp_lemma, pp_ts = pp_inp[1:indexsep-1],pp_inp[indexsep+1:end]
            if 0 < length(setdiff(p_ts,pp_ts)) < 3
                push!(data, (x=Int[specialIndicies.bow], xp=set[i], xpp=set[j]))
            end
        end
    end
    data, Set(inputs), Set(outputs)
end

io_to_line(vocab::Vocabulary{SCANDataSet}, input, output) =
    "IN: "*join(input," ")*" OUT: "*join(output," ")


function io_to_line(vocab::Vocabulary{SIGDataSet}, input, output)
    #tags = Set(setdiff(vocab.inpdict.toElement,vocab.outdict.toElement))
    indexsep = findfirst(t->t==specialTokens.iosep,input)
    isnothing(indexsep) && return nothing
    lemma, ts = input[1:indexsep-1], input[indexsep+1:end]
    "$(join(lemma))\t$(join(output))\t$(join(ts,';'))"
end


function print_samples(model, processed, esets; beam=true, fname="samples.txt", N=400, K=300)
    vocab  = model.vocab
    data, inputs, outputs = pickprotos(model, processed, esets)
    inpdict, outdict = vocab.inpdict.toIndex, vocab.outdict.toIndex
    printed = Set(String[])
    iter = Iterators.Stateful(Iterators.cycle(shuffle(data)))
    isfile(fname) && rm(fname)
    while length(printed) < N
        for s in sample(model, iter; N=128, prior=true, beam=true)
            if s.probs[1] > (1.2 / model.config["beam_width"])
                tokens = model.vocab.tokens[s.sampleenc]
                if "→" ∈ tokens && length(findall(d->d=="→", tokens)) == 1
                    index = findfirst(t->t=="→",tokens)
                    input,output = tokens[1:index-1], tokens[index+1:end]
                    (length(input) == 0 || length(output)==0) && continue
                    line = io_to_line(vocab, input, output)
                    if !isnothing(line) && line ∉ printed &&
                       join(input,' ') ∉ inputs &&
                       join(output,' ') ∉ outputs &&
                       all(i->haskey(inpdict,i), input) &&
                       all(i->haskey(outdict,i), output)
                        push!(printed,line)
                        open(fname, "a+") do f
                            println(f,line)
                        end
                        length(printed) == N && break
                    end
                end
            end
        end
    end
    return printed
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


function sample(model::Recombine, dataloader; N::Int=model.config["N"], sampler=sample, prior=true, beam=true, forviz=false)
    if !(dataloader isa Base.Iterators.Stateful)
        dataloader = Iterators.Stateful(dataloader)
    end
    B  = max(model.config["B"],128)
    vocab = model.vocab
    samples = []
    while true
        d = getbatch_recombine(model, dataloader, B)
        if isnothing(d)
             warning("sampled less variable than expected")
             break
        end
        x, xp, xpp, copymasks, unbatched = d
        b = length(first(unbatched))
        z, Txp, Txpp = encode(model, x, xp, xpp; prior=prior)
        if beam
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
            probs2D = softmax(hcat(probs...), dims=2)
            predstr = [join(ntuple(k->trim(preds[k][i,:], vocab),length(preds)),'\n')  for i=1:B]
            predenc = [trimencoded(preds[1][i,:]) for i=1:b]
        else
            y, preds   = decode(model, x.tokens, Txp, Txpp, z, (xp.tokens,xpp.tokens); sampler=sampler, training=false)
            predstr    = mapslices(x->trim(x,vocab), preds, dims=2)
            probs2D    = ones(b,1)
            predenc    = [trimencoded(preds[i,:]) for i=1:b]
        end
        for i=1:b
            @inbounds push!(samples, (target    = join(vocab.tokens[unbatched[1][i]],' '),
                            xp        = join(vocab.tokens[unbatched[2][i]],' '),
                            xpp       = join(vocab.tokens[unbatched[3][i]],' '),
                            sample    = predstr[i],
                            sampleenc = predenc[i],
                            probs     = probs2D[i,:]))

        end
        length(samples) >= N && break
    end
    samples[1:N]
end

function print_ex_samples(model::Recombine, data; beam=false)
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

# function set_plot_font(fpath="Symbola.ttf")
#     font_manager = Plots.PyPlot.matplotlib["font_manager"]
#     font_dirs = ["$(pwd())"]
#     font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
#     font_list = font_manager.createFontList(font_files)
#     @show font_list
#     push!(font_manager.fontManager.ttflist, font_list...)
#         Plots.PyCall.PyDict(Plots.PyPlot.matplotlib["rcParams"])["font.family"] ="sans-serif"
#     Plots.PyCall.PyDict(Plots.PyPlot.matplotlib["rcParams"])["font.sans-serif"] =["Symbola"]
# end
#
#
# function realindex(V::Integer, xp_tokens::AbstractVector{<:Integer}, xpp_tokens::AbstractVector{<:Integer}, index::Integer)
#     if index > V
#         if index > V + length(xp_tokens)
#             return xpp_tokens[index-V-length(xp_tokens)]
#         else
#             return xp_tokens[index-V]
#         end
#     else
#         return index
#     end
# end
#
# realindex(model::Recombine, xp::Matrix{Int}, xpp::Matrix{Int}, indices::Vector{Int}) =
#     map(ki->realindex(length(model.vocab.tokens), view(xp,ki[1],:), view(xpp,ki[1],:), ki[2]), enumerate(indices))
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
               "N"=>100,
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

sigmorphon_config = Dict(
              "model"=> Recombine,
              "lang"=>"turkish",
              "kill_edit"=>false,
              "attend_pr"=>0,
              "A"=>32,
              "H"=>512,
              "Z"=>16,
              "E"=>64,
              "B"=>16,
              "attdim"=>32,
              "Kpos" =>16,
              "concatz"=>true,
              "optim"=>Adam(lr=0.001, gclip=5.0),
              "kl_weight"=>0.0,
              "kl_rate"=> 0.05,
              "fb_rate"=>4,
              "N"=>100,
              "useprior"=>true,
              "aepoch"=>1, #20
              "epoch"=>15,  #40
              "Ninter"=>10,
              "pdrop"=>0.5,
              "calctrainppl"=>false,
              "Nsamples"=>300,
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
              "task"=>SIGDataSet,
              "patiance"=>6,
              "lrdecay"=>0.5,
              "conditional" => true,
              "split" => "medium",
              "splitmodifier" => "jump",
              "beam_width" => 4,
              "copy" => true,
              "writedrop" => 0.5,
              "outdrop" => 0.1,
              "attdrop" => 0.1,
              "outdrop_test" => false,
              "positional" => true,
              "masktags" => false
              )
               # function print_samples_sig(model, eset;beam=true, fname="samples.txt", N=400)
               #     data, train_inputs, train_outputs = pickprotos(model, eset)
               #     inpdict, outdict = model.vocab.inpdict.toIndex, model.vocab.outdict.toIndex
               #     samples = sample(model, data; N=N*300, sampler=argmax, prior=true, beam=beam)
               #     tags = Set(setdiff(model.vocab.inpdict.toElement,model.vocab.outdict.toElement))
               #     cnt = 0
               #     open(fname, "w+") do f
               #         for s in samples
               #              if s.probs[1] > (1.2 / model.config["beam_width"])
               #                 datum = model.vocab.tokens[s.sampleenc]
               #                 if "→" ∈ datum && length(findall(d->d=="→", datum)) == 1
               #                     tokens = datum
               #                     index = findfirst(t->t=="→",tokens)
               #                     input,output = tokens[1:index-1], tokens[index+1:end]
               #                     indexsep = findfirst(t->t ∈ tags,input)
               #                     lemma, ts = input[1:indexsep-1], input[indexsep+1:end]
               #                     (length(ts) == 0 || length(lemma) == 0 || length(output) == 0) && continue
               #                     line = "$(join(lemma))\t$(join(output))\t$(join(ts,';'))"
               #                     input, output = parseDataLine(line, model.vocab.parser)
               #                     if join(input,' ') ∉ train_inputs && join(output,' ') ∉ train_outputs && all(i->haskey(inpdict,i), input) &&  all(i->haskey(outdict,i), output)
               #                         println(f,line)
               #                         cnt+=1
               #                         if cnt == N
               #                             break
               #                         end
               #                     end
               #                 end
               #              end
               #         end
               #     end
               # end
