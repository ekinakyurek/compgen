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
    @show proc
    @show isfile(proc)
    if isfile(proc)
        processed, esets, vocab, embeddings = load_preprocessed_data(proc)
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
        # elseif task==SCANDataSet
        #     embeddings = initializeWordEmbeddings(100; wordsDict=v.tokens.toIndex)
        #     m     = MT(v, config; embeddings=embeddings)
        else
            embeddings = nothing
            m     = MT(v, config)
        end
        processed  = preprocess(m, esets...)
        save_preprocessed_data(proc, v, processed, esets, embeddings)
    end
    if length(processed) == 2
        trn, dev = splitdata(shuffle(processed[1]),[0.9,0.1])
        processed = [trn,processed[2],dev]
    end
    return processed, esets, m
end


function get_data_cond_model(augmented_data, config, esets=nothing, vocab=nothing)
    task    = config["task"]
    subtask = config["subtask"]
    MT      = config["condmodel"]
    proc  = prefix(task, config) * "_$(MT)_processesed.jld2"
    # @show proc
    # @show isfile(proc)
    # if isfile(proc)
    #     processed, esets, vocab, embeddings = load_preprocessed_data(proc)
    #     m  = MT(vocab, config; embeddings=embeddings)
    # else
        if esets == nothing
            files = rawfiles(task, config)
            p     = Parser{task}()
            sets  = map(f->parseDataFile(f,p), files)
            limitdev!(sets,config["pplnum"])
            vocab = Vocabulary(first(sets),p)
            esets = map(s->encode(s,vocab), sets)
        end
        if task == YelpDataSet
            embeddings = initializeWordEmbeddings(300; wordsDict=v.tokens.toIndex)
            m     = MT(vocab, config; embeddings=embeddings)
        else
            embeddings = nothing
            m     = MT(vocab, config)
        end
        processed  = preprocess(m, esets...)
    #     save_preprocessed_data(proc, vocab, processed, esets, embeddings)
    # end
    #augmented_part
    augmented = map(d->encode_cond(d,vocab),augmented_data)

    if length(processed) == 2
        trn, dev = splitdata(shuffle(processed[1]),[0.95,0.05])
        processed = [trn,processed[2],dev]
    end

    return augmented, processed, esets, m
end

function main(config, condconfig=nothing)
    println("Preprocessing Data & Initializing Model")
    exp_time = Dates.format(Dates.now(), "mm-dd_HH.MM")
    Knet.seed!(config["seed"])
    processed, esets, model = get_data_model(config)
    task, cond, vocab =  config["task"], config["conditional"], model.vocab
    MT  = config["model"]
    #words = [Set(map(s->join(vocab.tokens[xfield(task,s,cond)],' '),set)) for set in esets]
    #@show rand(processed[1])
    println("example: ",[vocab.tokens[x] for x in rand(processed[1]) if x isa Vector{Int}])
    KLterm,trnlen = 0,1
    if MT <: Recombine || MT <: ProtoVAE
        KLterm = kl_calc(model)
        if task == YelpDataSet
            trnlen = 15091715
            sampler = (beam=false,)
        elseif task == SIGDataSet
            trnlen  = length(esets[1])^2
            sampler = (mixsampler=true,beam=false)
        elseif task==SCANDataSet
            trnlen  = length(first(pickprotos(model, processed, esets)))
            sampler = (beam=true,)
        end
    end

    train!(model, processed[1]; dev=processed[end], trnlen=trnlen)
    #samples = sample(model,processed[2])
    print_ex_samples(model,processed[2]; sampler...) # to diagnosis

    println("Calculating test evaluations")
    au       = calc_au(model, processed[2])
    mi       = calc_mi(model, processed[2])
    testppl  = calc_ppl(model, processed[2]; trnlen=trnlen)
    if config["calc_trainppl"]
        println("Calculating train ppl")
        trainppl = calc_ppl(model, processed[1]; trnlen=trnlen)
    else
        trainppl = nothing
    end
    println("Calculating dict ppl")
    dictppl = calc_ppl(model, processed[end]; trnlen=trnlen)
    existsamples, interex = [],[]

    langstr    = task == SIGDataSet ? string("_lang_",config["lang"]) : ""
    modelstr   = get(config,"copy",false) ? string(MT,"_copy") : string(MT)
    hintsstr   = get(config,"hints",-1) > -1 ? string("_hints_",config["hints"]) : ""
    seedstr    = string("_seed_",get(config,"seed",0))
    splitstr   = string("_split_",config["split"])
    hashstr    = string("_hash_",hash(config))
    saveprefix = string("checkpoints/",task,"/",modelstr,langstr,splitstr,hintsstr,seedstr,hashstr)
    samplefile = saveprefix*"_samples.txt"
    println("generating and printing samples")
    nonexistsamples, augmented_data = print_samples(model, processed, esets; fname=samplefile, N=config["Nsamples"], Nsample=2config["Nsamples"],sampler...)
    ex_test_data = rand(processed[1],10) #[map(field->vocab.tokens[field],d) for d in rand(processed[1],10)]
    result = (ex_test_data=ex_test_data, existsamples=existsamples,
              nonexistsamples=nonexistsamples, homot=interex, au=au, mi=mi,
              testppl=testppl, trainppl=trainppl, dictppl=dictppl,
              logtrnlen=log(trnlen), KL=KLterm)

    # println("saving the model and samples")
    # KnetLayers.save(saveprefix * "_results.jld2", result)

    # println("Generating Interpolation Examples")
    # interex  = sampleinter(model, processed[1])

    open(saveprefix * ".config","w+") do f
         printConfig(f,config)
    end

    open("gen_results.txt", "a+") do f
          println(f, saveprefix, "\t", result.testppl, "\t", result.dictppl, "\t", result.logtrnlen,"\t", result.KL)
    end

    if !isnothing(condconfig)
        model = nothing; GC.gc(); KnetLayers.gc()
        augmented, cond_processed, cond_esets, mcond = get_data_cond_model(augmented_data, condconfig, esets, vocab)
        if config["paug"] != 0
            cond_trn_data = Iterators.cycle(MultiIter(shuffle(augmented),shuffle(cond_processed[1]),config["paug"]))
            config["n_epoch_batches"] = length(cond_processed[1]) / (1-config["paug"]) / config["B"]
        else
            cond_trn_data = [cond_processed[1]; augmented]
            cond_trn_data = Iterators.cycle(shuffle(cond_trn_data))

        end
        train!(mcond, cond_trn_data; dev=cond_processed[end])
        cond_eval_test_aug = calc_ppl(mcond, cond_processed[2])
        cond_eval_val_aug  = calc_ppl(mcond, cond_processed[3])
        # mcond = nothing; GC.gc(); KnetLayers.gc()
        # mcond = Seq2Seq(vocab, condconfig)


        # cond_trn_data = Iterators.cycle(shuffle(cond_processed[1]))
        # train!(mcond, cond_trn_data; dev=cond_processed[end])
        # cond_eval_test = calc_ppl(mcond, cond_processed[2])
        # cond_eval_val  = calc_ppl(mcond, cond_processed[3])
        cond_eval_test = cond_eval_val = "none"
        open(saveprefix * ".config","a+") do f
          println(f,condconfig)
        end

        open("cond_results.txt", "a+") do f
          println(f, saveprefix, "\t", cond_eval_test_aug , "\t", cond_eval_val_aug, "\t", cond_eval_test,"\t", cond_eval_val)
        end

        result = (augmented=augmented,
                 ctest_aug = cond_eval_test_aug,
                 cval_aug = cond_eval_val_aug,
                 ctest = cond_eval_test_aug,
                 cval = cond_eval_val_aug,
                 result...)
    end
    result
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


#
# function run_experiments_simple(baseconfig)
#     f = open("resultr.csv","w+")
#     println(f,"H,E,Z,LR,Kappa,maxnorm,writedrop,positional,gclip,ppl")
#     close(f)
#     exps = get_experiments_simple()
#     for (i,exp) in enumerate(reverse(exps))
#         o = copy(baseconfig)
#         o["H"], o["E"], o["Z"], lr , o["Kappa"], o["writedrop"], o["positions"], gclip  = exp
#         o["A"] = 2o["Z"]
#         o["optim"] = Adam(lr=lr, gclip=gclip)
#         ppl = main_simple(o)
#         f = open("resultr.csv","a+")
#         println(f,(exp..., ppl))
#         close(f)
#     end
# end

function main_proto(config)

end

function eval_mixed(proto, rnnlm, test; p=0.1, trnlen=1)
    proto_nllh, kl, mem, data, inds = train!(proto,test; eval=true, returnlist=true, trnlen=trnlen)
    proto_nllh2,_ = train!(proto,test; eval=true, returnlist=true, trnlen=trnlen, prior=true)
    proto_nllh_kl = [min(p1+kl,p2) for (p1,p2) in zip(proto_nllh,proto_nllh2)]
    sentences = [data[first(ind)] for ind in inds]
    rnn_nllh, _  = train!(rnnlm, sentences; eval=true, returnlist=true)
    nwords = sum(map(d->sum(d.x .> 4)+1, sentences))
    @show kl, mem, nwords, length(sentences)
    proto_llh  = -(proto_nllh_kl .+ mem)
    mixed_nllh = -logsumexp(hcat(proto_llh .+ log(p), -rnn_nllh .+ log(1-p)), dims=2)
    @show exp(sum(mixed_nllh)/nwords)
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



# recombine_turk = Dict("Z" => 16,"Nsamples" => 300,"calctrainppl" => false,
# "maxLength" => 45,"attend_pr" => 0,"calc_trainppl" => false,"attdim" => 128,"attdrop" => 0.1,
# "split" => "medium","kl_rate" => 0.05,"max_norm" => 10.0,"optim" => Adam(0.002, 0.9, 0.999, 1.0e-8, 0, 0.0, nothing, nothing),
# "pdrop" => 0.5,"useprior" => true,"writedrop" => 0.1,"lang" => "turkish",
# "lrdecay" => 0.5,"fb_rate" => 4,"Ninter" => 10,"gradnorm" => 325.5951611838534,
# "num_examplers" => 2,"H" => 512,"concatz" => true,"conditional" => true,
# "masktags" => false,"Kpos" => 16,"task" => SIGDataSet,"patiance" => 6,
# "Nlayers" => 2,"outdrop_test" => false,
# "rwritedrop" => 0.1,"kl_weight" => 0.0,"rpatiance" => 0,
# "model" => Recombine,"aepoch" => 1,"copy" => true,"B" => 16,"outdrop" => 0.1,"pplnum" => 1000,"kill_edit" => false,"eps" => 1.0,"dist_thresh" => 0.5,"max_cnt_nb" => 5,"epoch" => 15,"splitmodifier" => "jump","beam_width" => 4,"N" => 100,"activation" => ELU,"A" => 32,"E" => 128,"positional" => true,"Kappa" => 25,"authresh" => 0.1)

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
