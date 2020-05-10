include("util.jl")
include("parser.jl")
include("models.jl")
include("recombine.jl")
include("seqseqatt.jl")

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

function main(config, cond_config=nothing)
    println("Preprocessing Data & Initializing Model")
    exp_time = Dates.format(Dates.now(), "mm-dd_HH.MM")
    processed, esets, model = get_data_model(config)
    task, cond, vocab =  config["task"], config["conditional"], model.vocab
    MT  = config["model"]
    #words = [Set(map(s->join(vocab.tokens[xfield(task,s,cond)],' '),set)) for set in esets]
    @show rand(processed[1])
    #println("example: ",map(x->vocab.tokens[x],rand(processed[1])))
    if MT <: Recombine || MT <: ProtoVAE
        @show KLterm = kl_calc(model)
        if task == YelpDataSet
            trnlen = 15091715
        else
            trnlen = length(first(pickprotos(model, processed, esets)))
            @show trnlen
        end
    else
        KLterm = 0
        trnlen = 1
    end

    train!(model, processed[1]; dev=processed[end], trnlen=trnlen)
    #samples = sample(model,processed[2])
    print_ex_samples(model,processed[2]; beam=true) # to diagnosis

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
    langstr  = task == SIGDataSet ? string("_lang_",config["lang"],"_") : "_"
    modelstr = get(config,"copy",false) ? string(MT,"copy") : string(MT)
    saveprefix = string("checkpoints/",task,"/",modelstr,langstr,"split_",config["split"],"_",exp_time,"_",hash(config))
    samplefile = saveprefix*"_samples.txt"
    println("generating and printing samples")
    nonexistsamples, augmented_data = print_samples(model, processed, esets; beam=true, fname=samplefile, N=config["Nsamples"], subtask=config["subtask"])
    ex_test_data = rand(processed[1],10) #[map(field->vocab.tokens[field],d) for d in rand(processed[1],10)]
    result = (model=model, ex_test_data=ex_test_data, existsamples=existsamples,
              nonexistsamples=nonexistsamples, homot=interex, au=au, mi=mi,
              testppl=testppl, trainppl=trainppl, dictppl=dictppl,
              logtrnlen=log(trnlen), KL=KLterm)
    # println("saving the model and samples")
    # KnetLayers.save(saveprefix * "_results.jld2", result)
    # if task == SCANDataSet
    #     println("converting samples to json for downstream task")
    #     jfile = to_json(model.vocab.parser, samplefile)
    #     println("copying samples to downstream location")
    #     for i=0:9
    #         cp(jfile,"geca/exp/scan_jump/retrieval/composed.$(i).json"; force=true)
    #     end
    # elseif task == SIGDataSet
    #     files = rawfiles(task, config)
    #     lang, split = config["lang"], config["split"]
    #     datafolder  = "emnlp2018-imitation-learning-for-neural-morphology/tests/data/"
    #     write("$(datafolder)$(lang)-train-$(split)",read(`cat data/Sigmorphon/task1/all/turkish-train-medium $(samplefile)`))
    #     write("$(datafolder)$(lang)-dev",read(`cat $(files[2])`))
    # end
    #
    # existsamples =  (trnsamples  = samples[findall(s->in(s,words[1]), samples)],
    #                  tstsamples  = samples[findall(s->in(s,words[2]), samples)],
    #                  dictsamples = samples[findall(s->in(s,words[end]), samples)])
    #
    # nonexistsamples =  samples[findall(s->(!in(s, words[1]) &&
    #                                        !in(s, words[2]) &&
    #                                        !in(s, words[end])), samples)]
    # println("Generating Interpolation Examples")
    # interex  = sampleinter(model, processed[1])

    # open(saveprefix * ".config","w+") do f
    #     println(f,config)
    # end
    # if task == SIGDataSet
    #     open("sigresults.txt", "a+") do f
    #         println(f, saveprefix, "\t", result.testppl, "\t", result.dictppl, "\t", result.logtrnlen,"\t", result.KL)
    #     end
    # end
    augmented, cond_processed, cond_esets, mcond = get_data_cond_model(augmented_data, cond_config, esets, vocab)
    cond_trn_data = Iterators.cycle(MultiIter(augmented,cond_processed[1],cond_config["paug"]))
    train!(mcond, cond_trn_data; dev=cond_processed[end])
    cond_eval = calc_ppl(mcond, cond_processed[2])
    result = (mcond=mcond, augmented=augmented, cond_eval=cond_eval, result...)
    return result
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

function eval_mixed(proto, rnnlm, test; p=0.1, trnlen=1)
    proto_nllh, kl, mem, data, inds = train!(proto,test; eval=true, returnlist=true, trnlen=trnlen)
    sentences = [data[first(ind)] for ind in inds]
    rnn_nllh, _  = train!(rnnlm, sentences; eval=true, returnlist=true)
    nwords = sum(map(d->sum(d.x .> 4)+1, sentences))
    @show kl, mem, nwords, length(sentences)
    proto_llh  = -(proto_nllh .+ (kl+mem))
    mixed_nllh = -logsumexp(hcat(proto_llh .+ log(p), -rnn_nllh .+ log(1-p)), dims=2)
    @show exp(sum(mixed_nllh)/nwords)
    -mixed_nllh, proto_llh, -rnn_nllh, sentences
end




rnnlm_sig_config = Dict(
               "model"=> RNNLM,
               "lang"=>"turkish",
               "kill_edit"=>false,
               "attend_pr"=>0,
               "A"=>32,
               "H"=>512,
               "Z"=>16,
               "E"=>64,
               "B"=>4,
               "attdim"=>128,
               "concatz"=>true,
               "optim"=>Adam(lr=0.001),
               "kl_weight"=>0.0,
               "kl_rate"=> 0.05,
               "fb_rate"=>4,
               "N"=>10000,
               "useprior"=>true,
               "aepoch"=>1, #20
               "epoch"=>100,  #40
               "Ninter"=>10,
               "pdrop"=>0.4,
               "calctrainppl"=>false,
               "Nsamples"=>300,
               "pplnum"=>1000,
               "authresh"=>0.1,
               "Nlayers"=>2,
               "Kappa"=>25,
               "max_norm"=>10.0,
               "eps"=>1.0,
               "activation"=>ELU,
               "maxLength"=>45,
               "calc_trainppl"=>false,
               "num_examplers"=>2,
               "dist_thresh"=>0.6,
               "max_cnt_nb"=>10,
               "task"=>SIGDataSet,
               "patiance"=>8,
               "lrdecay"=>0.5,
               "conditional" => true,
               "split" => "medium",
               "splitmodifier" => "right",
               "beam_width" => 4,
               "writedrop" => 0.1,
               "condmodel"=>Seq2Seq,
               "subtask"=>"reinflection",
               "paug"=>0.3
               )

proto_sig_config = Dict(
               "model"=> ProtoVAE,
               "lang"=>"turkish",
               "kill_edit"=>false,
               "attend_pr"=>0,
               "A"=>32,
               "H"=>512,
               "Z"=>16,
               "E"=>64,
               "B"=>8,
               "attdim"=>128,
               "concatz"=>true,
               "optim"=>Adam(lr=0.001),
               "kl_weight"=>0.0,
               "kl_rate"=> 0.05,
               "fb_rate"=>4,
               "N"=>10000,
               "useprior"=>true,
               "aepoch"=>1, #20
               "epoch"=>15,  #40
               "Ninter"=>10,
               "pdrop"=>0.4,
               "calctrainppl"=>false,
               "Nsamples"=>300,
               "pplnum"=>1000,
               "authresh"=>0.1,
               "Nlayers"=>2,
               "Kappa"=>25,
               "max_norm"=>10.0,
               "eps"=>1.0,
               "activation"=>ELU,
               "maxLength"=>45,
               "calc_trainppl"=>false,
               "num_examplers"=>2,
               "dist_thresh"=>0.5,
               "max_cnt_nb"=>25,
               "task"=>SIGDataSet,
               "patiance"=>6,
               "lrdecay"=>0.5,
               "conditional" => true,
               "split" => "medium",
               "splitmodifier" => "right",
               "beam_width" => 4,
               "copy" => true,
               "writedrop" => 0.1,
               "attdrop" => 0.1,
               "insert_delete_att" =>false,
               "condmodel"=>Seq2Seq,
               "subtask"=>"reinflection",
               "paug"=>0.3
               )



proto_yelp_config = Dict(
               "model"=> ProtoVAE,
               "lang"=>"turkish",
               "kill_edit"=>false,
               "attend_pr"=>0,
               "A"=>256,
               "H"=>300,
               "Z"=>64,
               "E"=>300,
               "B"=>128,
               "attdim"=>128,
               "concatz"=>true,
               "optim"=>Adam(lr=0.001),
               "kl_weight"=>0.0,
               "kl_rate"=> 0.05,
               "fb_rate"=>4,
               "N"=>10000,
               "useprior"=>true,
               "aepoch"=>1, #20
               "epoch"=>8,  #40
               "Ninter"=>10,
               "pdrop"=>0.1,
               "calctrainppl"=>false,
               "Nsamples"=>100,
               "pplnum"=>1000,
               "authresh"=>0.1,
               "Nlayers"=>3,
               "Kappa"=>25,
               "max_norm"=>10.0,
               "eps"=>1.0,
               "activation"=>ELU,
               "maxLength"=>25,
               "calc_trainppl"=>false,
               "num_examplers"=>2,
               "dist_thresh"=>0.5,
               "max_cnt_nb"=>10,
               "task"=>YelpDataSet,
               "patiance"=>4,
               "lrdecay"=>0.5,
               "conditional" => false,
               "split" => "simple",
               "splitmodifier" => "right",
               "beam_width" => 4,
               "copy" => false,
               "writedrop" => 0.1,
               "attdrop" => 0.1,
               "insert_delete_att" =>false
               )

recombine_scan_config = Dict(
              "model"=> Recombine,
              "lang"=>"turkish",
              "kill_edit"=>false,
              "attend_pr"=>0,
              "A"=>32,
              "H"=>512,
              "Z"=>16,
              "E"=>64,
              "B"=>32,
              "attdim"=>128,
              "Kpos" =>16,
              "concatz"=>true,
              "optim"=>Adam(lr=0.002),
              "gradnorm"=>1.0,
              "kl_weight"=>0.0,
              "kl_rate"=> 0.05,
              "fb_rate"=>4,
              "N"=>100,
              "useprior"=>true,
              "aepoch"=>1, #20
              "epoch"=>8 ,  #40
              "Ninter"=>10,
              "pdrop"=>0.5,
              "calctrainppl"=>false,
              "Nsamples"=>300,
              "pplnum"=>1000,
              "authresh"=>0.1,
              "Nlayers"=>2,
              "Kappa"=>25,
              "max_norm"=>10.0,
              "eps"=>1.0,
              "activation"=>ELU,
              "maxLength"=>45,
              "calc_trainppl"=>false,
              "num_examplers"=>2,
              "dist_thresh"=>0.5,
              "max_cnt_nb"=>5,
              "task"=>SCANDataSet,
              "patiance"=>4,
              "lrdecay"=>0.5,
              "conditional" => true,
              "split" => "add_prim",
              "splitmodifier" => "jump",
              "beam_width" => 4,
              "copy" => true,
              "writedrop" => 0.5,
              "outdrop" => 0.7,
              "attdrop" => 0.0,
              "outdrop_test" => true,
              "positional" => true,
              "masktags" => false,
              "condmodel"=>Seq2Seq,
              "subtask"=>nothing,
              "paug"=>0.01,
              "seperate"=>true
              )



recombine_sig_config = Dict(
             "model"=> Recombine,
             "lang"=>"spanish",
             "kill_edit"=>false,
             "attend_pr"=>0,
             "A"=>32,
             "H"=>512,
             "Z"=>16,
             "E"=>64,
             "B"=>32,
             "attdim"=>128,
             "Kpos" =>16,
             "concatz"=>true,
             "optim"=>Adam(lr=0.002),
             "gradnorm"=>1.0,
             "kl_weight"=>0.0,
             "kl_rate"=> 0.05,
             "fb_rate"=>4,
             "N"=>100,
             "aepoch"=>1, #20
             "epoch"=>15,  #40
             "Ninter"=>10,
             "pdrop"=>0.5,
             "calctrainppl"=>false,
             "Nsamples"=>300,
             "pplnum"=>1000,
             "authresh"=>0.1,
             "Nlayers"=>2,
             "Kappa"=>1.0,
             "max_norm"=>1.0,
             "eps"=>1.0,
             "activation"=>ELU,
             "maxLength"=>45,
             "calc_trainppl"=>false,
             "num_examplers"=>2,
             "dist_thresh"=>0.5,
             "max_cnt_nb"=>5,
             "task"=>SIGDataSet,
             "patiance"=>6,
             "lrdecay"=>0.5,
             "conditional" => true,
             "split" => "medium",
             "splitmodifier" => "jump",
             "beam_width" => 4,
             "copy" => true,
             "writedrop" => 0.3,
             "outdrop" => 0.3,
             "attdrop" => 0.3,
             "outdrop_test" => true,
             "positional" => true,
             "masktags" => false,
             "condmodel"=>Seq2Seq,
             "rwritedrop"=>0.0,
             "rpatiance"=>0,
             "subtask"=>"analyses",
             "paug"=>0.1,
             "seperate"=>true,
             "path"=>"jacob/morph/"
             )


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
