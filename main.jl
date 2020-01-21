include("util.jl")
include("parser.jl")
include("models.jl")

function limitdev!(sets, limit=1000)
    if length(sets) == 3
        endp  = length(sets[3])
        start = min(endp,limit)
        splice!(sets[3],start:endp)
    end
end

function get_data_model(config)
    task  = config["task"]
    MT    = config["model"]
    proc  = prefix(task, config) * "_processesed.jld2"
    if isfile(proc)
        processed, esets, vocab = load_preprocessed_data(config)
        m  = MT(vocab, config)
    else
        files = rawfiles(task, config)
        p     = Parser{task}()
        sets  = map(f->parseDataFile(f,p), files)
        limitdev!(sets,config["pplnum"])
        v     = Vocabulary(first(sets),p)
        esets = map(s->encode(s,v), sets)
        m     = MT(v, config)
        processed  = preprocess(m, esets...)
        save_preprocessed_data(m, processed, esets)
    end
    return processed, esets, m
end

function main(config)
    println("Preprocessing Data & Initializing Model")
    processed, esets, model = get_data_model(config)
    vocab = model.vocab
    task, cond = config["task"], config["conditional"]
    words = [Set(map(s->join(vocab.tokens[xfield(task,s,cond)]),set)) for set in esets]
    @show rand(words[1])
    train!(model, processed[1]; dev=processed[end])
    samples = sample(model,first(processed))
    existsamples =  (trnsamples  = samples[findall(s->in(s,words[1]), samples)],
                     tstsamples  = samples[findall(s->in(s,words[2]), samples)],
                     dictsamples = samples[findall(s->in(s,words[end]), samples)])

    nonexistsamples =  samples[findall(s->(!in(s, words[1]) &&
                                           !in(s, words[2]) &&
                                           !in(s, words[end])), samples)]

    println("Generating Interpolation Examples")
    interex  = sampleinter(model, processed[1])
    println("Calculating test evaluations")
    au       = calc_au(model, processed[2])
    mi       = calc_mi(model, processed[2])
    testppl = calc_ppl(model, processed[2])
    if config["calc_trainppl"]
        println("Calculating train ppl")
        trainppl = calc_ppl(model, processed[1])
    else
        trainppl = nothing
    end
    println("Calculating dict ppl")
    dictppl = calc_ppl(model, processed[end])
    println("--DONE--")
    return (existsamples=existsamples, nonexistsamples=nonexistsamples, homot=interex, au=au, mi=mi,testppl=testppl, trainppl=trainppl, dictppl=dictppl)
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

default_config = Dict(
               "model"=>VAE,
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
               "aepoch"=>20, #20
               "epoch"=>40,  #40
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
               "task"=>SIGDataSet,
               "patiance"=>6,
               "lrdecay"=>0.5,
               "conditional" => false,
               "split" => "template",
               "splitmodifier" => "right"
               )

#
#
# function main(config)
#     println("Parsing Data")
#     vocab, train, test, dict, train_words, test_words, dict_words = get_data(config)
#     MT = eval(Meta.parse(config["model"]))
#     model = MT(vocab, config)
#     train!(model, train; dev=train[1:1000])
#     samples = sample(model, train)
#     existsamples =  (trnsamples = samples[findall(s->haskey(train_words,s), samples)],
#                      tstsamples = samples[findall(s->haskey(test_words,s), samples)],
#                      dictsamples = samples[findall(s->haskey(dict_words,s), samples)])
#
#     nonexistsamples =  samples[findall(s->(!haskey(train_words,s) &&
#                                            !haskey(test_words,s) &&
#                                            !haskey(dict_words,s)), samples)]
#
#     println("Generating Interpolation Examples")
#     interex  = sampleinter(model, train)
#     println("Calculating test evaluations")
#     au       = calc_au(model, test)
#     mi       = calc_mi(model, test)
#
#     testppl = calc_ppl(model, test)
#     if config["calc_trainppl"]
#         println("Calculating train ppl")
#         trainppl = calc_ppl(model, train)
#     else
#         trainppl = nothing
#     end
#     println("Calculating dict ppl")
#     if !isempty(dict)
#         dictppl = calc_ppl(model, dict)
#     else
#         dictppl = nothing
#     end
#     println("--DONE--")
#     return (existsamples=existsamples, nonexistsamples=nonexistsamples, homot=interex, au=au, mi=mi,testppl=testppl, trainppl=trainppl, dictppl=dictppl)
# end

#
# function get_data(config)
#     if config["model"] == "ProtoVAE"
#         if config["task"] == "SCAN"
#             get_prototype_scan_data(config)
#         else
#             get_prototype_data(config)
#         end
#     else
#         if config["task"] == "SCAN"
#             get_vae_scan_data(config)
#         else
#             get_vae_data(config)
#         end
#     end
# end
#
# function get_vae_data(config)
#     lang=config["lang"]
#     # Parse SIGMORPHON Train Split
#     train = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-train-high"))
#     # Get Vocabular
#     vocab = Vocabulary(train)
#     # Parse Test Split
#     test  = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-test"))
#
#     th  = lang[1:3]
#     #Word Lookup
#     train_words  = Dict((join(x.surface),true) for x in train)
#     test_words   = Dict((join(x.surface),true) for x in test)
#     combined     = combinedicts(train_words, test_words)
#
#     # Offline Dictionary
#     dictionary   = [parseDataLine(line) for line in  eachline("./data/unimorph/$(th)/$(th)") if line != ""]
#     dictionary   = filter(x->!haskey(combined,join(x.surface)) , dictionary)
#     unseen_words = Dict((join(x.surface),true) for x in dictionary)
#     dictionary   = dictionary[randperm(length(dictionary))[1:min(config["pplnum"],end)]]
#     return vocab, train, test,  dictionary, train_words, test_words, unseen_words
# end
#
# function get_vae_scan_data(config)
#     # Parse SCAN Train Split
#     train = map(parseSCANLine, eachline("/home/gridsan/eakyurek/git/datagen/data/SCAN/template_split/tasks_train_template_around_right.txt"))
#     # Get Vocabular
#     vocab = VocabularySCAN(train)
#     # Parse Test Split
#     test  = map(parseSCANLine, eachline("/home/gridsan/eakyurek/git/datagen/data/SCAN/template_split/tasks_test_template_around_right.txt"))
#     test  = test[randperm(length(test))[1:min(config["pplnum"],end)]]
#     #Input Lookup
#     mask = [specialTokens.mask]
#     train_words  = Dict((join([x.in;mask;x.out],' '),true) for x in train)
#     test_words   = Dict((join([x.in;mask;x.out],' '),true) for x in test)
#     #combined     = combinedicts(train_words, test_words)
#     return vocab, train, test, [], train_words, test_words, Dict()
# end
#
#
# function get_prototype_scan_data(config)
#     # Parse SIGMORPHON Train Split
#     config["lang"] = "SCAN"
#     lang, thresh, maxcnt, testl = config["lang"], config["dist_thresh"], config["max_cnt_nb"], config["pplnum"]
#
#     train = map(parseSCANLine, eachline("/home/gridsan/eakyurek/git/datagen/data/SCAN/template_split/tasks_train_template_around_right.txt"))
#     # Get Vocabular
#
#     # Parse Test Split
#     test  = map(parseSCANLine,  eachline("/home/gridsan/eakyurek/git/datagen/data/SCAN/template_split/tasks_test_template_around_right.txt"))
#
#     #Word Lookup
#     mask = [specialTokens.mask]
#     train_words  = Dict((join([x.in;mask;x.out],' '),true) for x in train)
#     test_words   = Dict((join([x.in;mask;x.out],' '),true) for x in test)
#     vocab = VocabularySCAN([train;test])
#     prefix="data/prototype/$lang"
#     files = prefix .* ["train",  "test"] .* ".tsv"
#     if !isfile(first(files))
#         get_neighbours_dict(collect(keys(train_words)),
#                             collect(keys(test_words)),
#                             thresh=thresh, maxcnt=maxcnt, lang=lang, prefix=prefix)
#     end
#
#     trn,tst = map(f->parse_prototype_file(f, vocab)[1], files)
#     return vocab, trn, tst, nothing, train_words, test_words, nothing
# end
#
# function get_prototype_data(config)
#     # Parse SIGMORPHON Train Split
#     lang, thresh, maxcnt, testl = config["lang"], config["dist_thresh"], config["max_cnt_nb"], config["pplnum"]
#
#     train = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-train-high"))
#     # Get Vocabular
#
#     # Parse Test Split
#     test  = map(parseDataLine, eachline("./data/Sigmorphon/task1/all/$(lang)-test"))
#
#     th  = lang[1:3]
#     #Word Lookup
#     train_words  = Dict((join(x.surface),true) for x in train)
#     test_words   = Dict((join(x.surface),true) for x in test)
#     combined     = combinedicts(train_words, test_words)
#
#     # Offline Dictionary
#     dictionary   = [parseDataLine(line) for line in  eachline("./data/unimorph/$(th)/$(th)") if line != ""]
#
#     vocab = Vocabulary([train;test;dictionary])
#     dictionary   = filter(x->!haskey(combined,join(x.surface)) , dictionary)
#     unseen_words = Dict((join(x.surface),true) for x in dictionary)
#     prefix="data/prototype/$lang"
#     files = prefix .* ["train", "dev", "test"] .* ".tsv"
#     if isfile(prefix * "train.tsv")
#         get_neighbours_dict(collect(keys(train_words)),
#                             collect(keys(test_words)),
#                             collect(Iterators.take(keys(unseen_words), testl));
#                             thresh=thresh, maxcnt=maxcnt, lang=lang, prefix=prefix)
#     end
#
#     trn,dev,tst = map(f->parse_prototype_file(f, vocab)[1], files)
#     return vocab, trn, dev, tst, train_words, test_words, unseen_words
# end
