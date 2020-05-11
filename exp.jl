include("src/util.jl")
include("src/parser.jl")
include("src/attention.jl")
include("src/vae.jl")
include("src/rnnlm.jl")
include("src/proto.jl")
include("src/recombine.jl")
include("src/seqseqatt.jl")


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
    if haskey(config,"path") && task == SIGDataSet
        return read_from_jacobs_format(config["path"], config)
    end
    proc  = prefix(task, config) * "_processesed.jld2"
    println("processed file: ",proc," exist: ", isfile(proc))
    if isfile(proc)
        processed, esets, vocab, embeddings = load_preprocessed_data(proc)
        model  = MT(vocab, config; embeddings=embeddings)
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
    MT       = config["model"]
    task     = config["task"]
    langstr  = task == SIGDataSet ? string("_lang_",config["lang"]) : ""
    modelstr = get(config,"copy",false) ? string(MT,"_copy") : string(MT)
    hintsstr = get(config,"hints",-1) > -1 ? string("_hints_",config["hints"]) : ""
    seedstr  = string("_seed_",get(config,"seed",0))
    splitstr = string("_split_",config["split"])
    hashstr  = string("_hash_",hash(config))
    string("checkpoints/",task,"/",modelstr,langstr,splitstr,hintsstr,seedstr,hashstr)
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
    train!(model, processed[1]; dev=processed[end], trnlen=trnlen)
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
    open(saveprefix * ".config","w+") do f
         printConfig(f,config)
         println(f,"Choice Cost: ",log(trnlen))
         println(f,"KL Cost: ",KLterm)
         println(f, "TEST: ", testppl)
         println(f, "VAL: ", valppl)
    end
    return saveprefix, (processed, esets, vocab)
end

function sample_from_generative_model(saveprefix, trndata=nothing)
    model  = KnetLayers.load(saveprefix * "model.jld2","model")
    config = model.config
    task   = config["task"]
    if isnothing(trndata)
        proc  = prefix(config["task"], config) * "_processesed.jld2"
        trndata = load_preprocessed_data(proc)
    end
    processed, esets, vocab = trndata
    MT = typeof(model)
    if MT <: Recombine || MT <: ProtoVAE
        if task == YelpDataSet
            sampler = (beam=false,)
        elseif task == SIGDataSet
            sampler = (mixsampler=true,beam=false)
        elseif task==SCANDataSet
            sampler = (beam=true,)
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
    model.config["n_epoch_batches"] = n_epoch_batches
    train!(model, trn_data; dev=processed[end])
    println("TEST EVALS")
    eval_test = calc_ppl(model, processed[2]; printeval=true)
    println("VAL EVALS")
    eval_val  = calc_ppl(model, processed[3]; printeval=true)
    modelfile = saveprefix*"condmodel.jld2"
    println("saving the conditional model to $modelfile")
    KnetLayers.save(modelfile,"model", model)
    open(saveprefix * ".condconfig","a+") do f
         printConfig(f,config)
         println(f,"TEST: ", eval_test)
         println(f,"VAL: ", eval_val)
    end
    saveprefix, trndata, augmentedstr
end

function main(config, condconfig=nothing; generate=true, baseline=true, usegenerated=false)
    Knet.seed!(config["seed"])
    if generate
        saveprefix, trndata = train_generative_model(config)
        _,trndata, augmentedstr = sample_from_generative_model(saveprefix, trndata)
        if !isnothing(condconfig)
            evaluate_cond_model(condconfig, saveprefix, trndata, augmentedstr)
        end
    else
        processed, esets, m, generated = get_data_model(config)
        trndata = (processed, esets, m.vocab)
        m=nothing
        saveprefix = getsaveprefix(config)
    end
    if baseline && !isnothing(condconfig)
        augmented = usegenerated ? generated : nothing
        println("using augmented $usegenerated , length=$(length(augmented))")
        evaluate_cond_model(condconfig, saveprefix*"_baseline", trndata, augmented)
    end
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

function parse_results(fname)
    results = readlines(fname)
    for result in results
           name, testppl, dictppl, logtrnlen, KL = split(result,'\t')
           ppl, ptokLoss, pinstLoss =  eval(Meta.parse(testppl))
           start =  findnext("/",name,findfirst("/",name)[1]+1)[1]
           modelend = findnext("_",name,start)[1]
           model_name = name[start+1:modelend-1]
           langstart = findnext("_",name,modelend+1)[1]
           langend   = findnext("_",name,langstart+1)[1]
           lang_name = name[langstart+1:langend-1]
           split_start = findnext("_",name,langend+1)[1]
           split_end   = findnext("_",name,split_start+1)[1]
           split_name = name[split_start+1:split_end-1]
           println(name,",",lang_name,",",split_name,",",model_name,",",ppl,",",ptokLoss,",",pinstLoss,",", KL,",",logtrnlen)
    end
end


function eval_mixed(proto, rnnlm, test; p=0.1, trnlen=1)
    proto_nllh, kl, mem, data, inds = train!(proto,test; eval=true, returnlist=true, trnlen=trnlen)
    proto_nllh2,_ = train!(proto,test; eval=true, returnlist=true, trnlen=trnlen, prior=true)
    proto_nllh_kl = [min(p1+kl,p2) for (p1,p2) in zip(proto_nllh,proto_nllh2)]
    sentences = [data[first(ind)] for ind in inds]
    rnn_nllh, _  = train!(rnnlm, sentences; eval=true, returnlist=true)
    nwords = sum(map(d->sum(d.x .> 4)+1, sentences))
    @show kl, mem, nwords, length(sentences)
    proto_llh  = -(proto_nllh .+ (kl+mem))
    proto_llh  = -(proto_nllh_kl .+ mem)
    mixed_nllh = -logsumexp(hcat(proto_llh .+ log(p), -rnn_nllh .+ log(1-p)), dims=2)
    @show exp(sum(mixed_nllh)/nwords)
    -mixed_nllh, proto_llh, -rnn_nllh, sentences
    ms = map(groupit,length.(inds))
    xys = vcat(([x y] for (x,y) in zip(-proto_nllh .- mem, -rnn_nllh))...)
    xy = plot(-200:20:-10,-200:20:-10)
    plt = scatter!(xys[:,1], xys[:,2]; markersize=ms, legend=false)
    title!("Loglikelihood NLM vs Proto")
    yaxis!("NLM")
    xaxis!("Neural-Editor")
    Plots.savefig("protonlm_copy_$(config["copy"]).pdf")
    -mixed_nllh, proto_llh, -rnn_nllh, sentences, data, inds
 end

function groupit(l)
   if l==1; 1
   elseif l<50 && l>1; 2
   elseif l<300 && l>=50; 3
   elseif l>=300; 4; end
end
