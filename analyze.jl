using Clustering

model_load_file = "checkpoints/SCANDataSet/Recombine_nproto_2_vae_true_template_seed_0model.jld2"
model = KnetLayers.load(model_load_file,"model")
config = model.config
processed, esets, _ = get_data_model(config)



function sample_with_z(model::Recombine, dataloader; N::Int=model.config["N"], sampler=argmax, prior=false, beam=true, forviz=false, mixsampler=false, temp=0.2)
    if !(dataloader isa Base.Iterators.Stateful)
        dataloader = Iterators.Stateful(dataloader)
    end
    B  = max(model.config["B"],128)
    vocab = model.vocab
    samples = []
    while true
        d = getbatch(model, dataloader, B)
        if isnothing(d)
             warning("sampled less variable than expected")
             break
        end
        x, protos, copymasks, ID, unbatched = d
        b = length(first(unbatched))
        z, Tprotos= encode(model, x, protos; ID=ID, prior=prior)
        if beam && !mixsampler # FIXME: You can do beam with mixsampler
            preds, probs, scores, outputs  = beam_decode(model, x.tokens, Tprotos, z; forviz=forviz)
            if forviz
                for i=1:b
                    @inbounds push!(samples,process_for_viz(vocab,
                    preds[1][i,:],
                    ntupel(k->protos[k].tokens[i,:],length(protos)),
                    ntuple(k->scores[k][:,i,:],length(protos)),
                    outputs[:,i,:]))
                end
                length(samples) >= N && break
                continue
            end
            probs2D = hcat(probs...)
            predstr = [join(ntuple(k->trim(preds[k][i,:], vocab),length(preds)),'\n')  for i=1:b]
            predenc = [trimencoded(preds[1][i,:]) for i=1:b]
        else
            _, preds, probs = decode(model, x.tokens, Tprotos, z; sampler=sampler, training=false, mixsampler=mixsampler, temp=temp)
            predstr  = mapslices(x->trim(x,vocab), preds, dims=2)
            probs2D  = reshape(probs,b,1)
            predenc  = [trimencoded(preds[i,:]) for i=1:b]
        end
        for i=1:b
            @inbounds push!(samples, (target    = join(vocab.tokens[unbatched[1][i]],' '),
                                        xp        = join(vocab.tokens[unbatched[2][i]],' '),
                                        xpp       = join(vocab.tokens[unbatched[3][i]],' '),
                                        sample    = predstr[i],
                                        sampleenc = predenc[i],
                                        probs     = probs2D[i,:],
                                        z = cpucopy(z[:,i])))

        end
        length(samples) >= N && break
    end
    samples[1:N]
end

samples = sample_with_z(model,shuffle(processed[1]); N=10000, beam=true)
zmatrix = hcat(map(s->s.z,samples)...)
Ncluster = 4
R = kmeans(zmatrix, Ncluster; maxiter=200, display=:iter)
a = assignments(R)

for i=1:Ncluster
    xp_len  = .0
    xpp_len = .0
    total = 0
    for s in samples[findall(j->j==i,a)]
        x  = split(s.target)
        xp = split(s.xp)
        xpp = split(s.xpp)

        xp_len += length(xp)
        xpp_len +=length(xpp)
        total += 1
    end
    @show i,xp_len/total, xpp_len/total
end
