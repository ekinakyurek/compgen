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
        processed, esets, vocab, embeddings = load_preprocessed_data(config)
        m  = MT(vocab, config; embeddings=embeddings)
    else
        files = rawfiles(task, config)
        p     = Parser{task}()
        sets  = map(f->parseDataFile(f,p), files)
        limitdev!(sets,config["pplnum"])
        v     = Vocabulary(first(sets),p)
        esets = map(s->encode(s,v), sets)
        if task == YelpDataSet
            embeddings = initializeWordEmbeddings(300; wordsDict=v.tokens.toIndex)
            m     = MT(v, config; embeddings=embeddings)
        else
            embeddings = nothing
            m     = MT(v, config)
        end
        processed  = preprocess(m, esets...)
        save_preprocessed_data(m, processed, esets, embeddings)
    end
    return processed, esets, m
end

function main(config)
    println("Preprocessing Data & Initializing Model")
    processed, esets, model = get_data_model(config)
    vocab = model.vocab
    task, cond = config["task"], config["conditional"]
    words = [Set(map(s->join(vocab.tokens[xfield(task,s,cond)],' '),set)) for set in esets]
    @show rand(words[1])
    train!(model, processed[1]; dev=processed[end])
    samples = sample(model,processed[2])
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
    testppl  = calc_ppl(model, processed[2])
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
                word   = join(vocab.tokens[w])
                println(f, word, '\t', Array(μ[:,i]))
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


yelp_config = Dict(
               "model"=> ProtoVAE2,
               "lang"=>"turkish",
               "kill_edit"=>false,
               "A"=>256,
               "H"=>256,
               "Z"=>64,
               "E"=>300,
               "B"=>128,
               "attdim"=>128,
               "concatz"=>true,
               "optim"=>Adam(lr=0.001, gclip=5.0),
               "kl_weight"=>0.0,
               "kl_rate"=> 0.05,
               "fb_rate"=>4,
               "N"=>10000,
               "useprior"=>true,
               "aepoch"=>1, #20
               "epoch"=>8,  #40
               "Ninter"=>10,
               "pdrop"=>0.4,
               "calctrainppl"=>false,
               "Nsamples"=>100,
               "pplnum"=>1000,
               "authresh"=>0.1,
               "Nlayers"=>2,
               "Kappa"=>15,
               "max_norm"=>10.0,
               "eps"=>1.0,
               "activation"=>ELU,
               "maxLength"=>25,
               "calc_trainppl"=>false,
               "num_examplers"=>2,
               "dist_thresh"=>0.5,
               "max_cnt_nb"=>10,
               "task"=>YelpDataSet,
               "patiance"=>6,
               "lrdecay"=>0.5,
               "conditional" => false,
               "split" => "simple",
               "splitmodifier" => "right"
               )

#
# default_config = Dict(
#                "model"=> ProtoVAE,
#                "lang"=>"turkish",
#                "kill_edit"=>false,
 #               "A"=>256,
#                "H"=>512,
#                "Z"=>16,
#                "E"=>32,
#                "B"=>16,
#                "concatz"=>true,
#                "optim"=>Adam(lr=0.004),
#                "kl_weight"=>0.0,
#                "kl_rate"=> 0.05,
#                "fb_rate"=>4,
#                "N"=>10000,
#                "useprior"=>true,
#                "aepoch"=>15, #20
#                "epoch"=>30,  #40
#                "Ninter"=>10,
#                "pdrop"=>0.4,
#                "calctrainppl"=>false,
#                "Nsamples"=>100,
#                "pplnum"=>1000,
#                "authresh"=>0.1,
#                "Nlayers"=>1,
#                "Kappa"=>25,
#                "max_norm"=>1.0,
#                "eps"=>0.1,
#                "activation"=>ELU,
#                "maxLength"=>20,
#                "calc_trainppl"=>false,
#                "num_examplers"=>2,
#                "dist_thresh"=>0.5,
#                "max_cnt_nb"=>10,
#                "task"=>SCANDataSet,
#                "patiance"=>6,
#                "lrdecay"=>0.5,
#                "conditional" => false,
#                "split" => "simple",
#                "splitmodifier" => "right"
#                )
#
#
#
# proto_config = Dict(
#               "model"=> ProtoVAE,
#               "lang"=>"spanish",
#               "kill_edit"=>false,
#               "A"=>256,
#               "H"=>256,
#               "Z"=>16,
#               "E"=>32,
#               "B"=>16,
#               "concatz"=>true,
#               "optim"=>Adam(lr=0.004),
#               "kl_weight"=>0.0,
#               "kl_rate"=> 0.05,
#               "fb_rate"=>4,
#               "N"=>10000,
#               "useprior"=>true,
#               "aepoch"=>1, #20
#               "epoch"=>30,  #40
#               "Ninter"=>10,
#               "pdrop"=>0.4,
#               "calctrainppl"=>false,
#               "Nsamples"=>100,
#               "pplnum"=>1000,
#               "authresh"=>0.1,
#               "Nlayers"=>2,
#               "Kappa"=>15,
#               "max_norm"=>1.5,
#               "eps"=>0.5,
#               "activation"=>ELU,
#               "maxLength"=>20,
#               "calc_trainppl"=>false,
#               "num_examplers"=>2,
#               "dist_thresh"=>0.5,
#               "max_cnt_nb"=>10,
#               "task"=>SCANDataSet,
#               "patiance"=>6,
#               "lrdecay"=>0.5,
#               "conditional" => false,
#               "split" => "simple",
#               "splitmodifier" => "right"
#               )
#
# proto_config2 = Dict(
#             "model"=> ProtoVAE,
#             "lang"=>"turkish",
#             "kill_edit"=>false,
#             "A"=>256,
#             "H"=>512,
#             "Z"=>16,
#             "E"=>32,
#             "B"=>16,
#             "concatz"=>true,
#             "optim"=>Adam(lr=0.004),
#             "kl_weight"=>0.0,
#             "kl_rate"=> 0.05,
#             "fb_rate"=>4,
#             "N"=>10000,
#             "useprior"=>true,
#             "aepoch"=>1, #20
#             "epoch"=>30,  #40
#             "Ninter"=>10,
#             "pdrop"=>0.4,
#             "calctrainppl"=>false,
#             "Nsamples"=>100,
#             "pplnum"=>1000,
#             "authresh"=>0.1,
#             "Nlayers"=>2,
#             "Kappa"=>25,
#             "max_norm"=>10,
#             "eps"=>0.5,
#             "activation"=>ELU,
#             "maxLength"=>20,
#             "calc_trainppl"=>false,
#             "num_examplers"=>2,
#             "dist_thresh"=>0.5,
#             "max_cnt_nb"=>10,
#             "task"=>SIGDataSet,
#             "patiance"=>6,
#             "lrdecay"=>0.5,
#             "conditional" => false,
#             "split" => "simple",
#             "splitmodifier" => "right"
#             )
#
# rnn_config = Dict(
#             "model"=> RNNLM,
#             "lang"=>"turkish",
#             "kill_edit"=>false,
#             "A"=>256,
#             "H"=>256,
#             "Z"=>16,
#             "E"=>32,
#             "B"=>16,
#             "concatz"=>true,
#             "optim"=>Adam(lr=0.004),
#             "kl_weight"=>0.0,
#             "kl_rate"=> 0.05,
#             "fb_rate"=>4,
#             "N"=>10000,
#             "useprior"=>true,
#             "aepoch"=>1, #20
#             "epoch"=>30,  #40
#             "Ninter"=>10,
#             "pdrop"=>0.4,
#             "calctrainppl"=>false,
#             "Nsamples"=>100,
#             "pplnum"=>1000,
#             "authresh"=>0.1,
#             "Nlayers"=>2,
#             "Kappa"=>15,
#             "max_norm"=>1.0,
#             "eps"=>0.5,
#             "activation"=>ELU,
#             "maxLength"=>20,
#             "calc_trainppl"=>false,
#             "num_examplers"=>2,
#             "dist_thresh"=>0.5,
#             "max_cnt_nb"=>10,
#             "task"=>SCANDataSet,
#             "patiance"=>6,
#             "lrdecay"=>0.5,
#             "conditional" => false,
#             "split" => "simple",
#             "splitmodifier" => "right"
#             )
