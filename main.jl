using Plots
import Plots: px

include("parser.jl")
include("models.jl")

function get_data(;lang="spanish")
    th = lang[1:3]
    data  = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-train-high"))
    vocab = Vocabulary(data)
    test  = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-test"))
    dictionary = [parseDataLine(line) for line in  eachline("./data/unimorph/$(th)/$(th)") if line != ""]
    trainsfs    = unique(map(x->join(x.surface),data))
    testsfs     = unique(map(x->join(x.surface),test))
    dictsfs     = unique([map(x->join(x.surface),dictionary); trainsfs; testsfs])
    unseensfs   = [x for x in dictsfs if x ∉ trainsfs]
    return vocab, data, test, trainsfs, testsfs, unseensfs
end

function main(lang="spanish", H=512, Z=16, E=16, B=16, concatz=true, optim=Adam(lr=0.002), epoch=40, kl_weight=0.0f0, kl_rate = 0.1f0, fb_rate=4, N=10000, useprior=true, aepoch=20, vaepoch=40, Ninter=10)
    vocab, data, test, trainsfs, testsfs, unseensfs = get_data(lang=lang)
    edata = encode(data,vocab)
    tdata = encode(test,vocab)
    morph2 = VAE(length(vocab.chars), edata.num; H=H, E=E, Z=Z, concatz=true)
    train_ae!(morph2, data, vocab; optim=Adam(), B=B, epoch=aepoch)
    morph3 = VAE(length(vocab.chars),  edata.num; H=H, E=E, Z=Z)
    copytoparams(morph3.encoder, morph2.encoder)
    copytoparams(morph3.Wμ, morph2.Wμ)
    copytoparams(morph3.Wσ, morph2.Wσ)
    copytoparams(morph3.dec_embed, morph3.encoder.embedding)
    morph2=nothing
    KnetLayers.gc()
    train_vae!(morph3, data, vocab; B=B, optim=optim, epoch=vaepoch, kl_weight=kl_weight, kl_rate = kl_rate, fb_rate=fb_rate)
    samples = sample(morph3, vocab, edata; N=N, useprior=true)  
    samples = (trsamples = samples[findall([s ∈ trainsfs for s in samples])], 
    tssamples = samples[findall([s ∈ testsfs for s in samples])],
    ussamples = samples[findall([s ∈ unseensfs for s in samples])])
    interex = sampleinter(morph3, vocab, data; N=Ninter)
    au, _, _ = calc_au(morph3, tdata; delta=0.01)
    mi       =  calc_mi(morph3,tdata)
    testppl  = calc_ppl(morph3, tdata; nsample=500, B=16)
    trainppl = calc_ppl(morph3, edata; nsample=500, B=16)
    return (model=morph3, samples=samples, homot=interex, au=au,mi=mi,testppl=testppl, trainppl=trainppl)
end





