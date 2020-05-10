struct SimpleAttention
    key_transform::Linear
    attdim::Int
end
function SimpleAttention(;memory::Int, query::Int, att::Int)
    SimpleAttention(Linear(input=memory,output=att, winit=att_weight_init),
                        att)
end

function (m::SimpleAttention)(memory, query; pdrop=0, mask=nothing)
    tquery  = query # [A,B,T] or [A,B]
    memory  = batch_last(memory)
    tkey    = m.key_transform(memory)
    pscores = nothing
    if ndims(query) == 2
        tquery = expand(tquery, dim=2)
    else
        tquery = batch_last(tquery)
    end
    scores = batch_last(bmm(tkey, tquery,transA=true)) #[T',B,T]
    scores = dropout(scores, pdrop)
    if !isnothing(mask)
        scores = applymask2(scores, expand(mask, dim=3), +) # size(mask) == [T', B , 1]
    end
    #memory H,B,T' # weights T' X B X T
    weights  = softmax(scores;dims=1) #[T',B,T]
    values   = batch_last(bmm(memory,batch_last(weights))) # [H,T,B]
    if  ndims(query) == 2
        mat(values, dims=1), mat(scores,dims=1), mat(weights, dims=1)
    else
        values, scores, weights
    end
end

struct Seq2Seq{Data}
    embed::Embed
    decoder::LSTM
    encoder::LSTM
    hidden_proj::Linear
    combine::Linear
    output::Linear
    attention::SimpleAttention
    copygate::Union{Dense,Nothing}
    vocab::Vocabulary{Data}
    config::Dict
end


function Seq2Seq(vocab::Vocabulary{T}, config; embeddings=nothing) where T<:DataSet
    V = length(vocab)
    copygate = config["copy"] ? Dense(input=V, output=1, activation=Sigm()) : nothing
    Seq2Seq{T}(load_embed(vocab, config, embeddings),
                 LSTM(input=config["E"]+config["H"], hidden=config["H"], numLayers=config["Nlayers"]),
                 LSTM(input=config["E"], hidden=config["H"], bidirectional=true, numLayers=config["Nlayers"]),
                 Linear(input=2config["H"], output=config["H"]),
                 Linear(input=2config["H"], output=config["H"]),
                 Linear(input=config["H"], output=V),
                 SimpleAttention(memory=config["H"], query=config["H"], att=config["H"]),
                 copygate,
                 vocab,
                 config)
end

function encode(model::Seq2Seq, packed)
    tokens = reshape(packed.tokens,length(packed.tokens),1)
    emb    = dropout(model.embed(tokens),model.config["pdrop"])
    out    = model.encoder(mat(emb,dims=1); batchSizes=packed.batchSizes, hy=true, cy=true)
    states = (out.hidden, out.memory)
    finals = ntuple(i->states[i][:,:,1] .+ states[i][:,:,2],2)
    #finals = source_hidden(x_context,x.lens)
    inds   = _batchSizes2indices(packed.batchSizes)
    x_context = PadRNNOutput2(model.hidden_proj(out.y), inds)
    finals, x_context
end

function copy_projection(vocab, tokens)
    proj = zeros(Float32, length(vocab), size(tokens,2), size(tokens,1))
    for i=1:size(tokens,1)
        for t=1:size(tokens,2)
            @inbounds proj[tokens[i, t], t, i] = 1
        end
    end
    return proj
end

const EPS = 1e-7
function decode_onestep(model::Seq2Seq, states, source, prev_context, input, copy_proj)
    pdrop, attdrop, outdrop = model.config["pdrop"], model.config["attdrop"], model.config["outdrop"]
    e  = mat(model.embed(input),dims=1)
    xi = dropout(vcat(e, prev_context[1]),pdrop)
    out = model.decoder(xi, states...; hy=true, cy=true)
    hbottom = out.y #vcat(out.y,e)
    s_attn, s_score, s_weight = model.attention(source.hiddens, hbottom; mask=source.mask) #positions[1]
    comb_features = dropout(model.combine(vcat(hbottom,s_attn)), pdrop)
    yfinal = model.output(comb_features)
    if model.config["copy"]
            pred_probs = logp(yfinal, dims=1)
            dist       = s_weight
            copy_probs = bmm(copy_proj, expand(dist,dim=2)) .+ EPS # V X T X B TIMES T X 1 X B
            copy_probs = log.(reshape(copy_probs,size(yfinal)))
            copy_weights = model.copygate(yfinal)
            comb_probs = cat((log.(copy_weights .+ EPS) .+ copy_probs),(log.((1+EPS) .- copy_weights) .+ pred_probs),dims=3)
            comb_probs = logsumexp(comb_probs, dims=3)
            pred_logits= comb_probs
    else
            pred_logits = logp(yfinal, dims=1)
            copy_weights = nothing
    end
    pred_logits, (comb_features,), (s_score,), (out.hidden, out.memory), (copy_weights,)
end

function decode(model::Seq2Seq, source_enc, source_finals, source_hiddens, copy_proj, target=nothing; sampler=argmax, training=true)
    T,B        = eltype(arrtype), size(source_enc.mask,1)
    H,V,E      = model.config["H"], length(model.vocab.tokens), model.config["E"]
    source     = (hiddens=source_hiddens, mask=arrtype(source_enc.mask'*T(-1e18)))
    input      = ones(Int,B,1) .* specialIndicies.bow
    states     = source_finals #map(f->_repeat(f,model.config["Nlayers"],dim=3), source_finals)
    limit      = (training ? size(target,2)-1 : model.config["maxLength"])
    preds      = ones(Int, B, limit)
    outputs    = []
    attns      = (zeroarray(arrtype,model.config["H"],B),)
    for i=1:limit
         output, attns, scores, states, c_weights = decode_onestep(model, states, source, attns, input, copy_proj)
         if !training
             i == 1 ? negativemask!(output,1:6) : negativemask!(output,1:2,4)
         end
         push!(outputs,output)
         if !training
             output        = convert(Array, exp.(output))
             preds[:,i]    = vec(mapslices(sampler, output, dims=1))
             input         = preds[:,i:i]
         else
             input         = target[:,i+1:i+1]
         end
    end
    return reshape(cat1d(outputs...),V,B,limit), preds
end

function beam_decode(model::Seq2Seq, source_enc, source_finals, source_hiddens, copy_proj; forviz=false)
    T,B        = eltype(arrtype), size(source_enc.mask,1)
    H,V,E, attdim = model.config["H"], length(model.vocab.tokens), model.config["E"], model.config["attdim"]
    source     = (hiddens=source_hiddens, mask=arrtype(source_enc.mask'*T(-1e18)))
    input      = ones(Int,B,1) .* specialIndicies.bow
    states     = map(f->_repeat(f,model.config["Nlayers"],dim=3), source_finals)
    input      = ones(Int,B,1) .* specialIndicies.bow #BOW
    attentions = (zeroarray(arrtype,H,B),)
    limit      = model.config["maxLength"]
    traces     = [(zeros(1, B), attentions, states, ones(Int, B, limit), input, nothing, nothing, nothing)]
    for i=1:limit
        traces = beam_search(model, traces, source, copy_proj; step=i, forviz=forviz)
    end
    outputs     = map(t->t[4], traces)
    probs       = map(t->vec(t[1]), traces)
    score_arr, output_arr, w_arr = traces[1][6], traces[1][7], traces[1][8]
    return outputs, probs, score_arr, output_arr, w_arr
end


function beam_search(model::Seq2Seq, traces, source, copy_proj; step=1, forviz=false)
    result_traces = []
    bw = model.config["beam_width"]
    @inbounds for i=1:length(traces)
        probs, attentions, states, preds, cinput, scores_arr, output_arr, w_arr = traces[i]
        y, attentions, scores, states, c_weights = decode_onestep(model, states, source, attentions, cinput, copy_proj)
        step == 1 ? negativemask!(y,1:6) : negativemask!(y,1:2,4)
        out_soft = convert(Array, exp.(y))
        output  = convert(Array,y)
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
            scores_softmaxed = map(s->Array(softmax(s,dims=1)), scores)
            copy_weights     = map(Array,c_weights)
        else
            scores_softmaxed = nothing
            copy_weights     = nothing
        end
        push!(result_traces, ((cprobs .+ probs), attentions, states, preds, inputs, scores_softmaxed, out_soft, scores_arr, output_arr, w_arr, copy_weights))
    end

    global_probs     = vcat(map(first, result_traces)...)
    global_srt_inds  = mapslices(x->sortperm(x; rev=true), global_probs, dims=1)[1:bw,:]
    global_srt_probs = sort(global_probs; rev=true, dims=1)[1:bw,:]
    new_traces = []
    for i=1:bw
        probs      = global_srt_probs[i:i,:]
        inds       = map(s->divrem(s,bw),global_srt_inds[i,:] .- 1)
        attentions = [hcat((result_traces[trace+1][2][1][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...)]
        states     = [cat((result_traces[trace+1][3][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds))..., dims=2) for k=1:2]
        inputs     = vcat((result_traces[trace+1][5][loc+1,bi] for (bi, (trace, loc)) in enumerate(inds))...)
        if step == 1
            old_preds  = copy(result_traces[1][4])
        else
            old_preds  = vcat((result_traces[trace+1][4][bi:bi,:] for (bi, (trace, _)) in enumerate(inds))...)
        end
        old_preds[:,step] .= inputs

        if forviz
            scores  = ntuple(k->hcat((result_traces[trace+1][6][k][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...),1)
            outsoft = hcat((result_traces[trace+1][7][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...)
            outsoft = reshape(outsoft,size(outsoft)...,1)
            scores  = map(s->reshape(s,size(s)...,1),scores)
            copy_weights = ntuple(k->hcat((result_traces[trace+1][11][k][:,bi:bi] for (bi, (trace, _)) in enumerate(inds))...),1)
            copy_weights = map(s->reshape(s,size(s)...,1),copy_weights)
            if step == 1
                scores_arr    = scores
                output_arr    = outsoft
                w_arr         = copy_weights
            else
                old_scores    = ntuple(k->cat([result_traces[trace+1][8][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2),1)
                old_weights   = ntuple(k->cat([result_traces[trace+1][10][k][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2),1)
                old_outputs   = cat([result_traces[trace+1][9][:,bi:bi,:] for (bi, (trace, _)) in enumerate(inds)]..., dims=2)
                scores_arr    = ntuple(i->cat(old_scores[i],scores[i],dims=3),1)
                w_arr         = ntuple(i->cat(old_weights[i],copy_weights[i],dims=3),1)
                output_arr    = cat(old_outputs,outsoft,dims=3)
            end
        else
            scores_arr, output_arr, w_arr = nothing, nothing, nothing
        end
        push!(new_traces, (probs,attentions,states,old_preds,old_preds[:,step:step], scores_arr, output_arr, w_arr))
    end
    return new_traces
end

function loss(model::Seq2Seq, data; eval=false, returnlist=false)
    source_enc, target, packed, unbatched = data
    yinds = (target.tokens[:, 2:end] .* target.mask[:, 2:end])
    ntokens = sum(target.mask[:, 2:end])
    if ntokens == 0; println("warning: empty batch"); end
    copy_proj = arrtype(copy_projection(model.vocab, source_enc.tokens))
    finals, hiddens = encode(model, packed)
    B = size(source_enc.mask,1)
    output, _ = decode(model, source_enc, finals, hiddens, copy_proj, target.tokens)
    if eval
        _, preds = decode(model, source_enc, finals, hiddens, copy_proj; training=false)
        preds = [trimencoded(preds[i,:], eos=true) for i=1:B]
        # preds,_ = beam_decode(model, source_enc, finals, hiddens)
        # preds = [trimencoded(preds[1][i,:]) for i=1:B]
    end
    if !returnlist
        inds = KnetLayers.findindices(output,yinds)
        loss = -sum(output[inds]) / ntokens
    else
        logpy = output
        loss = []
        for i=1:B
            inds = fill!(similar(yinds),0)
            inds[i,:] .= yinds[i,:]
            linds = KnetLayers.findindices(logpy, inds)
            push!(loss,-sum(logpy[linds]))
        end
        loss
    end
    if eval
        return loss, preds
    else
        return loss
    end

end


function getbatch(model::Seq2Seq, iter, B)
    edata = collect(Iterators.take(iter,B))
    b = length(edata); b==0 && return nothing
    unk, mask, eow, bow,_,_ = specialIndicies
    inputs, outputs = unzip(edata)
    inputs  = limit_seq_length(inputs; maxL=model.config["maxLength"])
    r   = sortperm(inputs, by=length, rev=true)
    inputs = inputs[r]
    outputs = outputs[r]
    input_packed = _pack_sequence(inputs)
    outputs = limit_seq_length_eos_bos(outputs; maxL=model.config["maxLength"])
    input_enc = PadSequenceArray(inputs, pad=mask, makefalse=true)
    output_enc = PadSequenceArray(outputs, pad=mask, makefalse=false)
    return (source=(input_enc..., lens=length.(inputs)), target=output_enc, packed=input_packed, unbatched=(inputs, outputs))
end


function viz(model::Seq2Seq, data; N=5)
    samples = sample(model, shuffle(data); N=N, beam=true, forviz=true)
    vocab = model.vocab.tokens
    #json = map(d->JSON.lower((x=vocab[d.x], xp=vocab[d.xp], xpp=vocab[d.xpp], scores1=d.scores[1], scores2=d.scores[2], probs=d.probs)), samples)
    #open("attention_maps.json", "w+") do f

    #    JSON.print(f,json, 4)
    #end
    for i=1:length(samples)
        x, scores, probs, source, target = samples[i]
        attension_visualize(model.vocab, probs, scores, source, x; prefix="$i")
        println("samples: ", join(model.vocab.tokens[x],' '))
        println("source: ", join(model.vocab.tokens[source],' '))
        println("target: ", join(model.vocab.tokens[target],' '))
        println("----------------------")
    end
end

function process_for_viz2(model::Seq2Seq, pred, source, scores, probs, w_arr, target)
    vocab = model.vocab
    xtrimmed    = trimencoded(pred)
    target      = trimencoded(target)
    xps         = (trimencoded(source),)
    attscores   = [score[1:length(xps[k]),1:length(xtrimmed)] for  (k,score) in enumerate(scores)]
    # ixp_end     = length(vocab)
    # indices     = collect(1:ixp_end)
    outsoft     = vcat(probs[:,1:length(xtrimmed)],w_arr[1][:,1:length(xtrimmed)])
    (x=xtrimmed, scores=attscores, probs=outsoft, xp=xps[1], target=target)
end

function attension_visualize(vocab, probs, scores, xp, x; prefix="")
    x  = vocab.tokens[x]
    y1 = vocab.tokens[xp]
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
    words = vocab.tokens.toElement
    y3 = [words; "COPY"]
    @show length(y3)
    @show size(probs,1)
    l = @layout [a b]
    p1 = heatmap(scores[1];yticks=(1:length(y1),y1),title="xp-x attention", attributes...)
    p3 = heatmap(probs;yticks=(1:length(y3),y3), title="action probs", attributes...)
    p  = Plots.plot(p3,p1; layout=l)
    Plots.savefig(p, prefix*"_attention_map.pdf")
end

function sample(model::Seq2Seq, data; N=nothing, sampler=argmax,beam=true, forviz=false)
    N  = isnothing(N) ? model.config["N"] : N
    B  = min(model.config["B"],32)
    dt = data
    vocab = model.vocab
    samples = []
    for i = 1 : (N ÷ B) + 1
        b =  min(N,B)
        if (d = getbatch(model,dt,b)) !== nothing
            source_enc, target, packed, unbatched = d
            copy_proj = arrtype(copy_projection(model.vocab, source_enc.tokens))
            finals, hiddens = encode(model, packed)
            B = size(source_enc.mask,1)
            if beam
                preds, probs, scores, outputs, w_arr  = beam_decode(model, source_enc, finals, hiddens, copy_proj; forviz=forviz)
                if forviz
                    for i=1:b
                        @inbounds push!(samples, process_for_viz2(model,
                        preds[1][i,:],
                        source_enc.tokens[i,:],
                        ntuple(k->scores[k][:,i,:],1),
                        outputs[:,i,:],
                        ntuple(k->w_arr[k][:,i,:],1),
                        target.tokens[i,:]
                        ))
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
                y, preds   = decode(model, source_enc, finals, hiddens, copy_proj; sampler=sampler, training=false)
                predstr    = mapslices(x->trim(x,vocab), preds, dims=2)
                probs2D    = ones(b,1)
                predenc    = [trimencoded(preds[i,:]) for i=1:b]
            end
            for i=1:b
                push!(samples, (target  = join(vocab.tokens[unbatched[2][i]],' '),
                                source  = join(vocab.tokens[unbatched[1][i]],' '),
                                sample  = predstr[i],
                                sampleenc = predenc[i],
                                probs=probs2D[i,:]))
            end
        end
    end
    return samples
end


function train!(model::Seq2Seq, data; eval=false, dev=nothing, dev2=nothing, returnlist=false)
    ppl = typemax(Float64)
    acc = 0.0
    if !eval
        bestparams = model.config["bestval"] ? deepcopy(parameters(model)) : nothing
        setoptim!(model,model.config["optim"])
        model.config["rpatiance"] = model.config["patiance"]
        dt = Iterators.Stateful(data)
        n_epoch_batches = model.config["n_epoch_batches"]
        n_epochs = model.config["n_epoch"]
    else
        dt = Iterators.Stateful(shuffle(data))
        n_epoch_batches = ((length(dt)-1) ÷ model.config["B"])+1
        n_epochs = 1
        losses = []
    end

    total_iter, ppl, ptokloss, pinstloss, acc, f1 = 0, .0, .0, .0, .0, .0
    for i=1:n_epochs
        lss, ntokens, ninstances, correct = .0, .0, .0, .0
        tp, fp, fn = .0, .0, .0
        #dt  = Iterators.Stateful((eval ? data : data)) #FIXME: SHUFFLE!
        msg(p) = string(@sprintf("Iter: %d,Lss(ptok): %.2f,Lss(pinst): %.2f", total_iter, lss/ntokens, lss/ninstances))
        #for i in progress(msg,1:n_epoch_batches)
        for i=1:n_epoch_batches
            d = getbatch(model,dt,model.config["B"])
            isnothing(d) && break
            b  = size(d[1].mask,1)
            n  = sum(d[2].mask[:,2:end])
            ntokens    += n
            ninstances += b
            total_iter += 1
            if !eval
                J = @diff loss(model, d)
                (isinf(value(J)) || iszero(value(J))) && continue

                if model.config["gradnorm"] > 0
                    clip_gradient_norm_(J, model.config["gradnorm"])
                end

                for w in parameters(J)
                    g = grad(J,w)
                    if !isnothing(g)
                        KnetLayers.update!(value(w), g, w.opt)
                    end
                end
                lss += (value(J)*n)
            else
                if returnlist
                    ls, preds = loss(model, d; eval=true, returnlist=true)
                    append!(losses,ls)
                    J = mean(ls)
                    lss += (J*b)
                else
                    J, preds = loss(model, d; eval=true, returnlist=false)
                    lss += (J*n)
                end
                toutputs = map(x->x[2:end],d.unbatched[2])
                correct += sum(preds .== toutputs)
                for i=1:b
                    pred_here, ref = preds[i], toutputs[i]
                    #println(join(model.vocab.tokens[pred_here],' '), join(model.vocab.tokens[ref],' '))
                    tp += length([p for p in pred_here if p ∈ ref])
                    fp += length([p for p in pred_here if p ∉ ref])
                    fn += length([p for p in ref if p ∉ pred_here])
                end
            end
            # if !eval && i%500==0
            #     print_ex_samples(model, data)
            # end
        end
        if !isnothing(dev)
            cur_ppl,_,_,cur_acc,cur_f1 = calc_ppl(model, dev)
            @show cur_f1, f1, cur_acc, acc
            if cur_acc > 0.4
                calc_ppl(model, cond_processed[2])
            end
            if cur_acc - acc > 1e-4
                if !isnothing(bestparams)
                    for (best,current) in zip(bestparams,parameters(model))
                        copyto!(value(best),value(current))
                    end
                end
                ppl = cur_ppl
                acc = cur_acc
                f1  = cur_f1
                model.config["rpatiance"]  = model.config["patiance"]
            else
                model.config["rpatiance"] = model.config["rpatiance"] - 1
                if model.config["rpatiance"] == 0
                     lrdecay!(model, model.config["lrdecay"])
                     model.config["rpatiance"] = model.config["patiance"]
                else
                    println("patiance decay, rpatiance: $(model.config["rpatiance"])")
                end
            end
        elseif !eval
            println((loss=lss/ntokens,))
        end
        if eval
            if returnlist
                return losses, data
            end
            ppl  = exp(lss/ntokens);
            acc  = correct/ninstances;
            prec = tp / (tp + fp)
            rec  = tp / (tp + fn)
            if prec == 0 || rec == 0
                f1 = 0
            else
                f1 = 2 * prec * rec / (prec + rec)
            end
            ptokloss=lss/ntokens
            pinstloss=lss/ninstances
            acc=correct/ninstances
        end
        #total_iter > 400000 && break
        # if total_iter % 30000 == 0
        #     GC.gc(); KnetLayers.gc()
        # end
    end
    if !isnothing(dev) && !eval && !isnothing(bestparams)
        for (best, current) in zip(bestparams,parameters(model))
            copyto!(value(current),value(best))
        end
    end
    return (ppl=ppl, ptokloss=ptokloss, pinstloss=pinstloss, acc=acc, f1=f1)
end

calc_ppl(model::Seq2Seq, dev) = train!(model, dev; eval=true)


import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength,SizeUnknown, @propagate_inbounds


struct MultiIter{I}
    iter1::I
    iter2::I
    p::Real #p from iter1, 1-p from iter2
end
import Random: shuffle
shuffle(m::MultiIter{I}) where I = MultiIter{I}(shuffle(m.iter1), shuffle(m.iter2),m.p)

IteratorSize(m::Type{MultiIter{I}}) where {I}   = SizeUnknown()
IteratorEltype(m::Type{MultiIter{I}}) where {I} = IteratorEltype(I)
function Iterators.cycle(m::MultiIter{I}) where {I}
     MultiIter(Iterators.cycle(m.iter1),Iterators.cycle(m.iter2),m.p)
end

function iterate(m::MultiIter, s...)
    iter = rand() < m.p ? (1,m.iter1) : (2, m.iter2)
    if length(s) == 0
        next = iterate(iter[2])
    elseif s[1][iter[1]] === nothing
        next = iterate(iter[2])
    else
        next = iterate(iter[2], s[1][iter[1]])
    end

    if next !== nothing
        (val, ss) = next
    else
        (val, ss) = nothing, nothing
    end

    if ss == nothing
        return nothing
    end

    if length(s) == 0
        nextmulti = ntuple(i->(i==iter[1] ? ss : nothing),2)
    else
        nextmulti = ntuple(i->(i==iter[1] ? ss : s[1][i]),2)
    end

    return (val,nextmulti)
end

function preprocess(model::Seq2Seq, train, devs...)
    map((train, devs...)) do set
        uset = unique(set)
        map(uset) do d
            xy_field(model.config["task"],d,model.config["subtask"])
        end
    end
end



scan_cond_config = Dict(
              "H"=>512,
              "E"=>64,
              "B"=>64,
              "attdim"=>512,
              "optim"=>Adam(lr=0.001),
              "gradnorm"=>1.0,
              "N"=>100,
              "epoch"=>150,
              "pdrop"=>0.5,
              "Nlayers"=>1,
              "activation"=>ELU,
              "maxLength"=>45,
              "task"=>SCANDataSet,
              "patiance"=>10,
              "lrdecay"=>0.5,
              "split" => "add_prim",
              "splitmodifier" => "jump",
              "beam_width" => 4,
              "copy" => false,
              "outdrop" => 0.0,
              "attdrop" => 0.0,
              "outdrop_test" => false,
              "positional" => false,
              "condmodel"=>Seq2Seq,
              "subtask"=>nothing,
              "paug"=>0.05,
              "model"=>Recombine,
              "conditional"=>true,
              "n_epoch_batches"=>32,
              "n_epoch"=>150,
              "bestval"=>false
              )


sig_cond_config = Dict(
            "H"=>512,
            "E"=>64,
            "B"=>16,
            "attdim"=>512,
            "optim"=>Adam(lr=0.002),
            "gradnorm"=>1.0,
            "lang"=>"spanish",
            "N"=>100,
            "epoch"=>150,
            "pdrop"=>0.5,
            "Nlayers"=>1,
            "activation"=>ELU,
            "maxLength"=>45,
            "task"=>SIGDataSet,
            "patiance"=>10,
            "lrdecay"=>0.9,
            "beam_width" => 4,
            "copy" => false,
            "outdrop" => 0.0,
            "attdrop" => 0.0,
            "outdrop_test" => false,
            "positional" => false,
            "condmodel"=>Seq2Seq,
            "subtask"=>"analyses",
            "split"=>"jacob",
            "paug"=>0.1,
            "model"=>Recombine,
            "conditional"=>true,
            "n_epoch_batches"=>65,
            "n_epoch"=>80,
            "bestval"=>true
            )
