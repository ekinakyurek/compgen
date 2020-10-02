include("src/util.jl")
include("src/parser.jl")
include("src/attention.jl")
if get(ENV,"RECOMB_TASK","SCAN") == "SCAN"
    include("src/recombine_scan.jl")
else # MORPH
    include("src/recombine_sig.jl")
end

include("src/seqseqatt.jl")

function limitdev!(sets, limit=2000)
    if length(sets) > 2
        endp  = length(sets[end])
        start = min(endp,limit)
        splice!(sets[end],start:endp)
    end
end

function get_data_model(config)
    task  = config["task"]
    MT    = config["model"]
    if haskey(config,"path") && task == SIGDataSet
        return read_from_jsons(defaultpath(config["task"])*"/", config)
    end
    proc  = defaultpath(config["task"]) * "/" *config["splitmodifier"] * ".jld2"
    println("processed file: ",proc," exist: ", isfile(proc))
    if isfile(proc)
        processed, esets, vocab, embeddings = load_preprocessed_data(proc)
        if get(config,"modeldir",nothing) == nothing
            model  = MT(vocab, config; embeddings=embeddings)
        else
            model  = KnetLayers.load(config["modeldir"],"model")
        end
    else
        files = rawfiles(task, config)
        parser = Parser{task}()
        sets  = map(f->parseDataFile(f,parser), files)
        limitdev!(sets,config["pplnum"])
        vocab = Vocabulary(first(sets),parser)
        esets = map(s->encode(s,vocab), sets)
        if task == YelpDataSet
            embeddings = initializeWordEmbeddings(300; wordsDict=vocab.tokens.toIndex)
            model = MT(vocab, config; embeddings=embeddings)
        else
            embeddings = nothing
            model = MT(vocab, config)
        end
        processed  = preprocess(model, esets...)
        save_preprocessed_data(proc, v, processed, esets, embeddings)
    end
    if config["nproto"] == 0
        processed = map(p->unique(d->d.x,p),processed)
    elseif  config["nproto"] == 1
        processed = map(p->unique(d->(d.x, d.xp),p),processed)
    end
    if length(processed) == 2
        trn, dev = splitdata(shuffle(processed[1]),[0.9,0.1])
        processed = [trn,processed[2],dev]
    end
    return processed, esets, model
end


function get_data_cond_model(config, vocab, esets, augmented_data=nothing)
    task    = config["task"]
    subtask = config["subtask"]
    MT      = config["condmodel"]
    if task == YelpDataSet
        embeddings = initializeWordEmbeddings(300; wordsDict=vocab.tokens.toIndex)
        model     = MT(vocab, config; embeddings=embeddings)
    else
        embeddings = nothing
        model     = MT(vocab, config)
    end
    processed  = preprocess(model, esets...)
    if !isnothing(augmented_data)
        augmented_data = map(d->encode_cond(d,vocab),augmented_data)
    end
    if length(processed) == 2
        trn, dev = splitdata(shuffle(processed[1]),[0.95,0.05])
        processed = [trn,processed[2],dev]
    end
    return processed, model, augmented_data
end

function getsaveprefix(config)
    modelstr = string(config["nproto"],"proto.vae.", !config["kill_edit"])
    task     = config["task"]
    langstr  = hintsstr = ""
    if task == SIGDataSet
        langstr  = config["lang"] * "/"
        hintsstr = string(".hints.",config["hints"])
        splitstr = ""
    else
        splitstr = string(".",config["splitmodifier"])
    end
    seedstr  = string(".seed.",config["seed"])
    # splitstr = string(".split.",config["split"])
    # hashstr  = string("_hash_",hash(config))
    string(CHECKPOINT_DIR,"/",task,"/",langstr,modelstr,splitstr,hintsstr,seedstr,".")
end

function train_generative_model(config)
    println("Preprocessing Data & Initializing Model")
    exp_time = Dates.format(Dates.now(), "mm-dd_HH.MM")
    processed, esets, model = get_data_model(config)
    task, cond, vocab =  config["task"], config["conditional"], model.vocab
    MT  = config["model"]
    println("example: ",[vocab.tokens[x] for x in rand(processed[1]) if x isa Vector{Int}])
    KLterm,trnlen = 0,1
    if MT <: Recombine || MT <: ProtoVAE
        @show KLterm = kl_calc(model)
        KLterm = kl_calc(model)
        if task == YelpDataSet
            trnlen = 15091715
        elseif task == SIGDataSet
            trnlen  = length(esets[1])^2
        elseif task==SCANDataSet
            trnlen  = length(first(pickprotos(model, processed, esets)))
        end
    end
    if get(config,"modeldir",nothing) == nothing
        train!(model, processed[1]; dev=processed[end], trnlen=trnlen)
    else
	println("mymodeldir: ", config["modeldir"])	
    end 
    print_ex_samples(model,processed[2]; beam=true) # to diagnosis
    println("Calculating test evaluations")
    au       = calc_au(model, processed[2])
    mi       = calc_mi(model, processed[2])
    testppl  = calc_ppl(model, processed[2]; trnlen=trnlen)
    println("Calculating val evaluations")
    valppl  = calc_ppl(model, processed[end]; trnlen=trnlen)
    saveprefix = getsaveprefix(config)
    modelfile = saveprefix*"model.jld2"
    println("saving the model to $modelfile")
    KnetLayers.save(modelfile, "model", model)
    open(saveprefix * "config","w+") do f
         printConfig(f,config)
         println(f,"Choice Cost: ",log(trnlen))
         println(f,"KL Cost: ",KLterm)
         println(f, "TEST: ", testppl)
         println(f, "VAL: ", valppl)
    end
    return saveprefix, (processed, esets, vocab), model
end

function sample_from_generative_model(saveprefix, trndata=nothing; model=nothing)
    if isnothing(model)
        model  = KnetLayers.load(saveprefix * "model.jld2","model")
    end
    config = model.config
    task   = config["task"]
    if isnothing(trndata)
        if task == SIGDataSet && haskey(config,"path")
            trndata = read_from_jsons(defaultpath(config["task"])*"/", config)
        else
            proc  = defaultpath(config["task"]) * "/" *config["splitmodifier"] * ".jld2"
            trndata = load_preprocessed_data(proc)
        end
    end
    processed, esets, vocab = trndata
    MT = typeof(model)
    if MT <: Recombine || MT <: ProtoVAE
        if task == YelpDataSet
            sampler = (beam=false,)
        elseif task == SIGDataSet
            sampler = (mixsampler=true,beam=false)
        elseif task==SCANDataSet
            sampler = (mixsampler=!config["beam"],beam=config["beam"])
        end
    end
    samplefile = saveprefix*"samples.txt"
    println("generating and printing samples to $samplefile")
    nonexistsamples, augmented_data = print_samples(model, processed, esets; fname=samplefile, N=config["N"], Nsample=config["Nsamples"], sampler...)
    saveprefix, (processed, esets, vocab), augmented_data
end

function evaluate_cond_model(config, saveprefix, trndata, augmentedstr=nothing)
    _, esets, vocab = trndata
    processed, model, augmented = get_data_cond_model(config, vocab, esets, augmentedstr)
    if !isnothing(augmented)
        for (x,y) in augmented
            println("inp: ", vocab.tokens[x], "\t out: ", vocab.tokens[y])
        end
        paug = config["paug"]
        if paug != 0
            iter = MultiIter(shuffle(augmented),shuffle(processed[1]),paug)
            trn_data  = Iterators.cycle(iter)
            n_epoch_batches = (length(processed[1]) / (1-paug)) ÷ config["B"]
        else
            iter = [processed[1]; augmented]
            trn_data = Iterators.cycle(shuffle(iter))
            n_epoch_batches= ((length(iter)-1) ÷ config["B"])+1
        end
    else
        iter = processed[1]
        trn_data = Iterators.cycle(shuffle(iter))
        n_epoch_batches = ((length(iter)-1) ÷ config["B"])+1
    end
    # println("total data length: ", length(iter))
    model.config["n_epoch_batches"] = n_epoch_batches
    train!(model, trn_data; dev=processed[end])
    println("TEST EVALS")
    eval_test = calc_ppl(model, processed[2]; printeval=true)
    println("VAL EVALS")
    eval_val  = calc_ppl(model, processed[end]; printeval=true)
    modelfile = saveprefix*"condmodel.jld2"
    println("saving the conditional model to $modelfile")
    KnetLayers.save(modelfile,"model", model)
    open(saveprefix * "condconfig","w") do f
         printConfig(f,config)
         println(f,"TEST: ", eval_test)
         println(f,"VAL: ", eval_val)
    end
    if length(processed) > 3
        println("TEST EASY")
        eval_test_easy = calc_ppl(model, processed[3]; printeval=true)
        println("VAL EASY")
        eval_val_easy  = calc_ppl(model, processed[4]; printeval=true)
        open(saveprefix * "cond_easy_config","w") do f
             printConfig(f,config)
             println(f,"TEST_EASY: ", eval_test_easy)
             println(f,"VAL_EASY: ",  eval_val_easy)
        end
    end
    saveprefix, trndata, augmentedstr
end

function main(config, condconfig=nothing; generate=true, baseline=true, usegenerated=false, saveprefix=nothing)
    Knet.seed!(config["seed"])
    if generate
        saveprefix, trndata, model = train_generative_model(config)
        _,trndata, augmentedstr = sample_from_generative_model(saveprefix, trndata; model=model)
        model = nothing; GC.gc(); KnetLayers.gc()
        if !isnothing(condconfig)
            evaluate_cond_model(condconfig, saveprefix, trndata, augmentedstr)
        end
    else
        if isnothing(saveprefix)
            processed, esets, m, generated = get_data_model(config)
            trndata = (processed, esets, m.vocab)
            m=nothing
            saveprefix = getsaveprefix(config)
        else
            processed, esets, m = get_data_model(config)
            trndata = (processed, esets, m.vocab)
            m=nothing
            task   = config["task"]
            parser = Parser{task}()
            augmented_data  = parseDataFile(saveprefix*"samples.txt",parser)
            generated = map(x->xy_field(task,x,condconfig["subtask"]),augmented_data)
        end
        saveprefix = getsaveprefix(config)
    end
    if baseline && !isnothing(condconfig)
        if usegenerated
            augmented = generated
            println("using augmented $usegenerated , length=$(length(augmented))")
            condsuffix = "augmented"
        else
            augmented = nothing
            println("BASELINE RUNNING")
            condsuffix = "baseline"
        end
        evaluate_cond_model(condconfig, saveprefix*condsuffix, trndata, augmented)
    end
end
