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
    tquery  = m.query_transform(query) # [A,N,B]

    pscores = nothing

    if !isnothing(positions) && ndims(tquery) == 2
        pscores2D = At_mul_B(m.aij_k(positions), tquery) # A x T' times A x B
        pscores   = reshape(pscores2D, 1, size(pscores2D,1), size(pscores2D,2)) # 1 X T' X B
    end

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

    if isnothing(pscores)
        score = dropout(bmm(tquery,tkey,transA=true),pdrop) ./ sqrt(m.attdim) #[1,T',B]
    else
        score = dropout(bmm(tquery,tkey,transA=true) .+ pscores, pdrop) ./ sqrt(m.attdim) #[1,T',B]
    end

    if !isnothing(mask)
        score = applymask(score, mask, +)
    end
    #memory H,B,T'
    weights  = softmax(score;dims=2) #[1,T',B]
    values   = mat(bmm(memory,weights;transB=true)) # [H,B]
    if !isnothing(pscores)
        pmemory = m.aij_v(positions) # H X T'
        values = pmemory * mat(weights) + mat(values)
    end
    return values, mat(score)
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
    aij_k = Embed(att_weights(input=2*16+1, output=config["attdim"]))
    aij_v = Embed(param(config["H"],2*16+1,; atype=arrtype))
    dinput = 2config["H"] + config["E"]  + config["A"]
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


function beam_decode(model::Recombine, x, xp, xpp, z, protos)
    T,B         = eltype(arrtype), size(z,2)
    H,V,E       = model.config["H"], length(model.vocab.tokens), model.config["E"]
    contexts    = (batch_last(xp.context), batch_last(xpp.context))
    masks       = (arrtype(xp.mask'*T(-1e18)), arrtype(xpp.mask'*T(-1e18)))
    input       = x[:,1:1] #BOW
    attentions  = (zeroarray(arrtype,H,B), zeroarray(arrtype,H,B))
    states      = (expand_hidden(model.h0,B), expand_hidden(model.c0,B))
    limit       = model.config["maxLength"]
    traces      = [(zeros(1, B), attentions, states, ones(Int, B, limit), input)]
    for i=1:limit
        traces = beam_search(model, traces, contexts, masks, z, protos; step=i)
    end
    outputs     = map(t->t[4], traces)
    probs       = map(t->vec(t[1]), traces)
    return outputs, probs
end

function cat_copy_scores(m::Recombine, y, scores)
    if model.config["copy"]
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

function position_calc(L, step, K)
    (map(x->abs(x) > K ? sign(x) * K : x,collect(0:L-1).-(step-1)) .+ K) .+ 1
end

function beam_search(model::Recombine, traces, contexts, masks, z, protos; step=1)
    result_traces = []
    bw = model.config["beam_width"]
    for i=1:length(traces)
        probs, attentions, states, preds, cinput = traces[i]
        positions = map(p->position_calc(size(p,2), step, 16), protos)
        y, attentions, scores, states      = decode_onestep(model, attentions, states, contexts, masks, z, cinput, positions)
        if step == 1
            y[1:5,:]  .= -1.0f18
        else
            y[1:2,:]  .= -1.0f18; y[4,:]    .= -1.0f18
        end
        yp                                 = cat_copy_scores(model, y, scores)
        output                             = convert(Array, softmax(yp,dims=1))
        #output                             = sumprobs(output, protos[1], protos[2])
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
        push!(result_traces, ((cprobs .+ probs), attentions, states, preds, inputs))
    end
    global_probs     = vcat(map(first, result_traces)...)
    global_srt_inds  = mapslices(x->sortperm(x; rev=true), global_probs, dims=1)[1:bw,:]
    global_srt_probs = sort(global_probs; rev=true, dims=1)[1:bw,:]
    new_traces = []
    for i=1:bw
        probs      = global_srt_probs[i:i,:]
        inds       = map(s->divrem(s,bw),global_srt_inds[i,:] .- 1)
        attentions = [hcat((result_traces[trace+1][2][k][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...) for k=1:2]
        states     = [cat((result_traces[trace+1][3][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds))..., dims=2) for k=1:2]
        inputs     = vcat((result_traces[trace+1][5][loc+1,bi] for (bi, (trace, loc)) in enumerate(inds))...)
        if step == 1
            old_preds  = copy(result_traces[1][4])
        else
            old_preds  = vcat((result_traces[trace+1][4][bi:bi,:] for (bi, (trace, _)) in enumerate(inds))...)
        end
        old_preds[:,step] .= realindex(model, protos[1], protos[2], inputs)
        #old_preds[:,step] .= inputs
        push!(new_traces, (probs,attentions,states,old_preds,old_preds[:,step:step]))
    end
    return new_traces
end


function decode(model::Recombine, x, xp, xpp, z, protos; sampler=sample, training=true)
    T,B        = eltype(arrtype), size(z,2)
    H,V,E      = model.config["H"], length(model.vocab.tokens), model.config["E"]
    contexts   = (batch_last(xp.context), batch_last(xpp.context))
    masks      = (arrtype(xp.mask'*T(-1e18)),  arrtype(xpp.mask'*T(-1e18)))
    input      = x[:,1:1] #BOW
    attentions = (zeroarray(arrtype,H,B), zeroarray(arrtype,H,B))
    states     = (expand_hidden(model.h0,B), expand_hidden(model.c0,B))
    limit      = (training ? size(x,2)-1 : model.config["maxLength"])
    preds      = ones(Int, B, limit)
    outputs    = []
    for i=1:limit
         positions = map(p->position_calc(size(p,2), i, 16), protos)
         y, attentions, scores, states = decode_onestep(model, attentions, states, contexts, masks, z, input, positions)
         if !training
             if i == 1
                 y[1:5,:]  .= -1.0f18
             else
                 y[1:2,:]  .= -1.0f18; y[4,:] .= -1.0f18
             end
         end
         output   = cat_copy_scores(model, y, scores)
         push!(outputs, output)
         if !training
             preds[:,i]    = realindex(model, protos..., vec(mapslices(sampler, convert(Array, value(output)),dims=1)))
             input         = preds[:,i:i]
         else
             input         = x[:,i+1:i+1]
         end
    end
    Tp = sum(s->size(s,2),protos)
    return reshape(cat1d(outputs...),V+Tp,B,limit), preds
end

function decode_onestep(model::Recombine, attentions, states, contexts, masks, z, input, positions)
    pdrop = model.config["pdrop"]
    e                         = mat(model.embed(input),dims=1)
    xi                        = vcat(e, attentions[1], attentions[2], z)
    out                       = model.decoder(xi, states[1], states[2]; hy=true, cy=true)
    h, c                      = out.hidden, out.memory
    hbottom                   = vcat(h[:,:,1],z)
    xp_attn, xp_score         = model.xp_attention(contexts[1],hbottom;mask=masks[1], pdrop=0.1, positions=positions[1])
    xpp_attn, xpp_score       = model.xp_attention(contexts[2],hbottom;mask=masks[2], pdrop=0.1, positions=positions[2])
    y                         = model.output(vcat(dropout(out.y, 0.7; drop=true),xp_attn,xpp_attn))
    yv                        = At_mul_B(model.embed.weight,y)
    yv, (xp_attn, xpp_attn), (xp_score, xpp_score), (h,c)
end

function encode(m::Recombine, x, xp, xpp; prior=false)
    pdrop =  m.config["pdrop"]
    xp_context    = m.xp_encoder(dropout(m.embed(xp.tokens), pdrop)).y
    xpp_context   = m.xp_encoder(dropout(m.embed(xpp.tokens), pdrop)).y
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
    ((findall(b->b==i, write_batches), findall(b->b==i, copy_batches)) for i=1:BT)
end

function getvalues(arr, values)
    if isempty(values)
        return 0f0
    else
        return sum(arr[values])
    end
end

# function loss(model::Recombine, data; average=false)
#     x, xp, xpp, copyinds, unbatched = data
#     B = length(first(unbatched))
#     copy_indices  = cat1d(copyinds...)
#     z, xp_context, xpp_context = encode(model, x, xp, xpp; prior=false)
#     Txp  = (mask=xp.mask, context=xp_context)
#     Txpp = (mask=xpp.mask, context=xpp_context)
#     y, _ = decode(model, x.tokens, Txp, Txpp, z, (xp.tokens, xpp.tokens))
#     write_indices = findindices(y, (x.tokens[:, 2:end] .* x.mask[:, 2:end]), dims=1)
#     probs = softmax(mat(y, dims=1), dims=1, algo=1)
#     write_probs = probs[write_indices]
#     copy_probs  = probs[copy_indices]
#     ind2bt = ind2BT(write_indices, copy_indices, size(probs)...)
#     -sum((log(write_probs[first(w)] + getvalues(copy_probs,c)) for (w,c) in ind2bt if !isempty(w))) / B
# end


function loss(model::Recombine, data; average=false)
    x, xp, xpp, copyinds, unbatched = data
    B = length(first(unbatched))
    copy_indices  = cat1d(copyinds...)
    z, xp_context, xpp_context = encode(model, x, xp, xpp; prior=false)
    Txp  = (mask=xp.mask, context=xp_context)
    Txpp = (mask=xpp.mask, context=xpp_context)
    y, _ = decode(model, x.tokens, Txp, Txpp, z, (xp.tokens, xpp.tokens))
    write_indices = findindices(y, (x.tokens[:, 2:end] .* x.mask[:, 2:end]), dims=1)
    nlly = logp(mat(y, dims=1), dims=1)
    -sum(nlly[[write_indices;copy_indices]])/B
end

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


function sample(model::Recombine, data; N=nothing, sampler=sample, prior=true, sanitize=false, beam=false)
    N  = isnothing(N) ? model.config["N"] : N
    B  = min(model.config["B"],32)
    dt = Iterators.Stateful(shuffle(data))
    vocab = model.vocab
    samples = []
    for i = 1 : (N ÷ B) + 1
        b =  min(N,B)
        if (d = getbatch_recombine(model, dt,b)) !== nothing
            x, xp, xpp, copymasks, unbatched = d
            B = length(first(unbatched))
            z, xp_context, xpp_context = encode(model, x, xp, xpp; prior=prior)
            Txp  = (mask=xp.mask, context=xp_context)
            Txpp = (mask=xpp.mask, context=xpp_context)
            if beam
                preds, probs   = beam_decode(model, x.tokens, Txp, Txpp, z, (xp.tokens,xpp.tokens))
                probs2D = softmax(hcat(probs...), dims=2)
                s       = hcat(map(pred->vec(mapslices(x->trim(x,vocab), pred, dims=2)), preds)...)
                s       = mapslices(s->join(s,'\n'),s,dims=2)
            else
                y, preds   = decode(model, x.tokens, Txp, Txpp, z, (xp.tokens,xpp.tokens); sampler=sampler, training=false)
                s          = mapslices(x->trim(x,vocab), preds, dims=2)
            end
            for i=1:b
                push!(samples, (target  = join(vocab.tokens[unbatched[1][i]],' '),
                                xp      = join(vocab.tokens[unbatched[2][i]],' '),
                                xpp     = join(vocab.tokens[unbatched[3][i]],' '),
                                sample  = s[i],
                                probs   = probs2D[i,:] ))
            end
        end
    end
    return samples
end


limit_seq_length(x; maxL=30) = map(s->(length(s)>maxL ? s[1:Int(maxL)] : s) , x)

function getbatch_recombine(model, iter, B)
    edata   = collect(Iterators.take(iter,B))
    mask    = specialIndicies.mask
    V       = length(model.vocab.tokens)
    if (b = length(edata)) != 0
        d           = (x, xp1, xp2) = unzip(edata)
        xp,xpp      = rand()>0.5 ? (xp1,xp2) : (xp2,xp1)
        x           = limit_seq_length(x) # FIXME: maxlength as constant
        xp          = map(s->[s;specialIndicies.eow], limit_seq_length(xp))
        xpp         = map(s->[s;specialIndicies.eow], limit_seq_length(xpp))
        seq_xp      = (tokens=PadSequenceArray(xp, pad=mask),  mask=get_mask_sequence(length.(xp); makefalse=true), lengths=length.(xp))
        seq_xpp     = (tokens=PadSequenceArray(xpp, pad=mask),  mask=get_mask_sequence(length.(xpp); makefalse=true), lengths=length.(xpp))
        seq_x       = (tokens=PadSequenceArray(map(xi->[specialIndicies.bow;xi;specialIndicies.eow], x), pad=mask),
                       mask=get_mask_sequence(length.(x) .+ 2; makefalse=false),
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
    end
    total_iter = 0
    for i=1:(eval ? 1 : model.config["epoch"])
        lss, ntokens, ninstances = 0.0, 0.0, 0.0
        dt  = Iterators.Stateful(shuffle(data))
        msg(p) = string(@sprintf("Iter: %d,Lss(ptok): %.2f,Lss(pinst): %.2f", total_iter, lss/ntokens, lss/ninstances))
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
            else
                J = loss(model, d; average=false)
            end
            lss        += (value(J)*b)
            ntokens    += n
            ninstances += b
            if !eval && i%100==0
                print_ex_samples(model, data; beam=true)
                if !isnothing(dev)
                    print_ex_samples(model, dev; beam=true)
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
        total_iter > 400000 && break
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
               "B"=>16,
               "attdim"=>32,
               "concatz"=>true,
               "optim"=>Adam(lr=0.001, gclip=10.0),
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
               "maxLength"=>30,
               "calc_trainppl"=>false,
               "num_examplers"=>2,
               "dist_thresh"=>0.5,
               "max_cnt_nb"=>10,
               "task"=>SCANDataSet,
               "patiance"=>10,
               "lrdecay"=>0.5,
               "conditional" => true,
               "split" => "add_prim",
               "splitmodifier" => "jump",
               "beam_width" => 5,
               "copy" => true
               )

function vmfKL(m::Recombine)
   k, d = m.config["Kappa"], 2m.config["Z"]
   k*((besseli(d/2.0+1.0,k) + besseli(d/2.0,k)*d/(2.0*k))/besseli(d/2.0, k) - d/(2.0*k)) +
   d * log(k)/2.0 - log(besseli(d/2.0,k)) -
   loggamma(d/2+1) - d * log(2)/2
end
