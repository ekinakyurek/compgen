include("util.jl")
include("parser.jl")
include("models2.jl")

# DEV SPLIT?
# DECIDE ON TEST SPLIT?
function get_data(config)
    if config["model"] == "ProtoVAE"
        get_prototype_data(config)
    else
        get_vae_data(config)
    end
end

function get_vae_data(config)
    lang=config["lang"]
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
    dictionary   = dictionary[randperm(length(dictionary))[1:min(config["pplnum"],end)]]
    return vocab, train, test,  dictionary, train_words, test_words, unseen_words
end

function get_prototype_data(config)
    # Parse SIGMORPHON Train Split
    lang, thresh, maxcnt, testl = config["lang"], config["dist_thresh"], config["max_cnt_nb"], config["pplnum"]

    train = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-train-high"))
    # Get Vocabular

    # Parse Test Split
    test  = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-test"))

    th  = lang[1:3]
    #Word Lookup
    train_words  = Dict((join(x.surface),true) for x in train)
    test_words   = Dict((join(x.surface),true) for x in test)
    combined     = combinedicts(train_words, test_words)

    # Offline Dictionary
    dictionary   = [parseDataLine(line) for line in  eachline("./data/unimorph/$(th)/$(th)") if line != ""]

    vocab = Vocabulary([train;test;dictionary])
    dictionary   = filter(x->!haskey(combined,join(x.surface)) , dictionary)
    unseen_words = Dict((join(x.surface),true) for x in dictionary)
    prefix="data/prototype/$lang"
    files = prefix .* ["train", "dev", "test"] .* ".tsv"
    if isfile(prefix * "train.tsv")
        get_neighbours_dict(collect(keys(train_words)),
                            collect(keys(test_words)),
                            collect(Iterators.take(keys(unseen_words), testl));
                            thresh=thresh, maxcnt=maxcnt, lang=lang, prefix=prefix)
    end

    trn,dev,tst = map(f->parse_prototype_file(f, vocab)[1], files)
    return vocab, trn, dev, tst, train_words, test_words, unseen_words
end

function printlatent(fname, model, data, vocab; B=16)
    edata = Iterators.Stateful(data)
    while ((d = getbatch(edata,B)) !== nothing)
        μ, logσ² = encode(model, d[1], isencatt(model) ? d[2] : nothing)
        inds  = _batchSizes2indices(d[1].batchSizes)
        words = map(ind->d[1].tokens[ind], inds)
        open(fname,"a+") do f
            for (i,w) in enumerate(words)
                word   = join(vocab.chars[w])
                latent =  μ[:,i]
                println(f,word,'\t',Array(latent))
            end
        end
    end
end

function printsamples(fname,words)
    open(fname,"a+") do f
        for w in words
            println(f,w)
        end
    end
end

# Add parameter for decoder layers
# Test and Optimize RNNLM
# Test and Optimize Prototype Model
default_config = Dict(
               "model"=>"ProtoVAE",
               "lang"=>"turkish",
               "H"=>512,
               "Z"=>16,
               "E"=>32,
               "B"=>16,
               "concatz"=>true,
               "optim"=>Adam(lr=0.003),
               "kl_weight"=>0,
               "kl_rate"=> 0.1,
               "fb_rate"=>4,
               "N"=>10000,
               "useprior"=>true,
               "aepoch"=>20,
               "epoch"=>30,
               "Ninter"=>10,
               "pdrop"=>0.4,
               "calctrainppl"=>false,
               "Nsamples"=>100,
               "pplnum"=>1000,
               "authresh"=>0.1,
               "Nlayers"=>1,
               "Kappa"=>25,
               "max_norm"=>7.5,
               "eps"=>0.01,
               "activation"=>ELU,
               "maxLength"=>20,
               "calc_trainppl"=>false,
               "num_examplers"=>2,
               "dist_thresh"=>0.6,
               "max_cnt_nb"=>10,
               "patiance"=>6,
               "lrdecay"=>0.5
               )



function main(config)
    println("Parsing Data")
    vocab, train, test, dict, train_words, test_words, dict_words = get_data(config)
    MT = eval(Meta.parse(config["model"]))
    model = MT(vocab, config)
    train!(model, train; dev=test)
    samples = sample(model, train)
    existsamples =  (trnsamples = samples[findall(s->haskey(train_words,s), samples)],
                     tstsamples = samples[findall(s->haskey(test_words,s), samples)],
                     dictsamples = samples[findall(s->haskey(dict_words,s), samples)])

    nonexistsamples =  samples[findall(s->(!haskey(train_words,s) &&
                                           !haskey(test_words,s) &&
                                           !haskey(dict_words,s)), samples)]

    println("Generating Interpolation Examples")
    interex  = sampleinter(model, train)
    println("Calculating test evaluations")
    au       = calc_au(model, test)
    mi       = calc_mi(model, test)

    testppl = calc_ppl(model, test)
    if config["calc_trainppl"]
        println("Calculating train ppl")
        trainppl = calc_ppl(model, train)
    else
        trainppl = nothing
    end
    println("Calculating dict ppl")

    dictppl = calc_ppl(model, dict)
    println("--DONE--")
    return (existsamples=existsamples, nonexistsamples=nonexistsamples, homot=interex, au=au, mi=mi,testppl=testppl, trainppl=trainppl, dictppl=dictppl)
end
