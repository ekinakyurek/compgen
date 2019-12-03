include("util.jl")
include("parser.jl")
include("models.jl")

# DEV SPLIT?
# DECIDE ON TEST SPLIT?
function get_data(;lang="spanish")
    # Parse SIGMORPHON Train Split
    train = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-train-high"))
    # Get Vocabular
    vocab = Vocabulary(train)
    # Parse Test Split
    test  = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-test"))

    th  = lang[1:3]
    #Word Lookup
    train_words  = Dict((join(x.surface),true) for x in train)
    test_words   = Dict((join(x.surface),true) for x in test)
    combined     = combinedicts(train_words, test_words)

    # Offline Dictionary
    dictionary   = [parseDataLine(line) for line in  eachline("./data/unimorph/$(th)/$(th)") if line != ""]
    dictionary   = filter(x->!haskey(combined,join(x.surface)) , dictionary)
    unseen_words = Dict((join(x.surface),true) for x in dictionary)
    return vocab, train, test,  dictionary, train_words, test_words, unseen_words
end

function get_prototype_data(;lang="spanish", thresh=0.6, maxcnt=10, testl=1000)
    # Parse SIGMORPHON Train Split
    train = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-train-high"))
    # Get Vocabular
    vocab = Vocabulary(train)
    # Parse Test Split
    test  = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-test"))

    th  = lang[1:3]
    #Word Lookup
    train_words  = Dict((join(x.surface),true) for x in train)
    test_words   = Dict((join(x.surface),true) for x in test)
    combined     = combinedicts(train_words, test_words)

    # Offline Dictionary
    dictionary   = [parseDataLine(line) for line in  eachline("./data/unimorph/$(th)/$(th)") if line != ""]
    dictionary   = filter(x->!haskey(combined,join(x.surface)) , dictionary)
    unseen_words = Dict((join(x.surface),true) for x in dictionary)


    get_neighbours_dict(collect(keys(train_words)),
                        collect(keys(test_words)),
                        collect(Iterators.take(keys(unseen_words), testl));
                        thresh=thresh, maxcnt=maxcnt, lang=lang)
end


# Add parameter for decoder layers
# Test and Optimize RNNLM
# Test and Optimize Prototype Model
function main(modelType=:VAE;
               lang="spanish",
               H=512, Z=16, E=16, B=16,
               concatz=true,
               optim=Adam(lr=0.002),
               kl_weight=0.0f0,
               kl_rate = 0.1f0,
               fb_rate=4,
                N=10000,
               useprior=true,
               aepoch=20,
               epoch=40,
               Ninter=10,
               pdrop=0.4,
               calctrainppl=false,
               Nsamples=500,
               pplnum=1000,
               authresh=0.1)

    println("Parsing Data")
    vocab, train, test, dict, train_words, test_words, dict_words = get_data(lang=lang)
    etrain, etest, edict = encode(train,vocab), encode(test,vocab), encode(dict, vocab)

    if modelType != :LSTM_LM
        println("Training Auto Encoder")
        premodel = VAE(length(vocab.chars), etrain.num; H=H, E=E, Z=Z, concatz=concatz, pdrop=pdrop)
        train_ae!(premodel, train, vocab; optim=optim, B=B, epoch=aepoch)
        encoder, Wμ, Wσ = premodel.encoder, premodel.Wμ, premodel.Wσ
        premodel=nothing; GC.gc(); gpugc()
        println("Initializing VAE and Transfering Weights")
        model = VAE(length(vocab.chars),  etrain.num; H=H, E=E, Z=Z, pdrop=pdrop)
        transferto!(model.encoder, encoder)
        transferto!(model.Wμ, Wμ)
        transferto!(model.Wσ, Wσ)
        transferto!(model.dec_embed, encoder.embedding)
        encoder, Wμ, Wσ = nothing, nothing, nothing; GC.gc(); gpugc()
        println("Training VAE")
        train_vae!(model, train, vocab; B=B, optim=optim, epoch=epoch, kl_weight=kl_weight, kl_rate = kl_rate, fb_rate=fb_rate)
        println("Generating Samples")
        samples = sample(model, vocab, etrain; N=N, useprior=useprior)
    else
        model = LSTM_LM(length(vocab.chars); H=H, E=E)
        train_rnnlm!(model, train, vocab; epoch=epoch, optim=optim, B=B)
        println("Generating Samples")
        samples = samplelm(model, vocab; N=N, B=B)
    end

    existsamples =  (trnsamples = samples[findall(s->haskey(train_words,s), samples)],
                     tstsamples = samples[findall(s->haskey(test_words,s), samples)],
                     dictsamples = samples[findall(s->haskey(dict_words,s), samples)])

    nonexistsamples =  samples[findall(s->(!haskey(train_words,s) &&
                                           !haskey(test_words,s) &&
                                           !haskey(dict_words,s)), samples)]
    if modelType != :LSTM_LM
        println("Generating Interpolation Examples")
        interex  = nothing #sampleinter(model, vocab, train; N=Ninter)
        println("Calculating test evaluations")
        au = nothing#au, _, _ = calc_au(model, etest; delta=authresh,B=B)
        mi = calc_mi(model, etest; B=B)
    else
        au,mi,interex = nothing, nothing, nothing
    end

    if modelType != :LSTM_LM
        testppl = calc_ppl(model, etest, vocab; nsample=Nsamples, B=B)
        if calctrainppl
            println("Calculating train ppl")
            trainppl = calc_ppl(model, etrain, vocab; nsample=Nsamples, B=B)
        else
            trainppl = nothing
        end
        println("Calculating dict ppl")
        edict   = encode(dict[randperm(length(dict))[1:min(pplnum,end)]],vocab)
        dictppl = calc_ppl(model, edict, vocab; nsample=Nsamples, B=B)
        println("--DONE--")
    else
        testppl = calc_ppllm(model, etest, vocab; B=B)
        if calctrainppl
            println("Calculating train ppl")
            trainppl = calc_pplm(model, etrain, vocab; B=B)
        else
            trainppl = nothing
        end
        println("Calculating dict ppl")
        edict   = encode(dict[randperm(length(dict))[1:min(pplnum,end)]],vocab)
        dictppl = calc_ppllm(model, edict, vocab; B=B)
        println("--DONE--")
    end
    return (existsamples=existsamples, nonexistsamples=nonexistsamples, homot=interex, au=au, mi=mi,testppl=testppl, trainppl=trainppl, dictppl=dictppl)
end
