using ArgParse
include("exp.jl")

function run(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    s.description = "SCANDataSet Experiments"
    @add_arg_table! s begin
        ("--seed"; arg_type=Int; default=3; help="seed")
        ("--config"; arg_type=String; help="generative model's config file")
        ("--condconfig"; arg_type=String; help="conditional model's config file")
        ("--epoch"; arg_type=Int; default=8; help="epoch")
        ("--H"; arg_type=Int; default=512; help="hidden dim")
        ("--E"; arg_type=Int; default=64; help="embedding dim")
        ("--Nlayers"; arg_type=Int; default=1; help="number of rnn layers")
        ("--B"; arg_type=Int; default=32; help="batch size")
        ("--attdim"; arg_type=Int; default=128; help="attention dim")
        ("--gradnorm"; arg_type=Float64; default=1.0; help="global gradient norm clip, 0 for none")
        ("--writedrop"; arg_type=Float64; default=0.5; help="write dropout in copy")
        ("--outdrop"; arg_type=Float64; default=0.7; help="output dropout in rnn")
        ("--pdrop"; arg_type=Float64; default=.5; help="dropout in various locations")
        ("--attdrop"; arg_type=Float64; default=.0; help="attention dropout in attentions ")
        ("--optim"; arg_type=String; default="Adam(;lr=0.002)")
        ("--split"; arg_type=String; default="template")
        ("--splitmodifier"; arg_type=String; default="around_right")
        ("--copy"; action=:store_true; help="copy meachanism in rnn")
        ("--outdrop_test"; action=:store_true; help="dropout on output at test time")
        ("--paug"; arg_type=Float64; default=0.01; help="augmentation ratio for condtional model, 0 for direct concatenation")
        ("--baseline"; action=:store_true; help="run conditional model without augmentation")
        ("--generate"; action=:store_true; help="train generative model")
        ("--seperate_emb"; action=:store_true; help="seperate embeddings for input-output")
        ("--beam"; action=:store_true; help="beam search")
        ("--usegenerated"; action=:store_true; help="run conditional model without augmentation")
        ("--N"; arg_type=Int; default=400; help="number of rnn layers")
        ("--Nsamples"; arg_type=Int; default=500; help="number of rnn layers")
        ("--loadprefix"; help="sample file prefix")
        ("--nproto"; arg_type=Int; default=2; help="number of prototypes, 0,1,2")
        ("--use_insert_delete"; action=:store_true; help="use insert and delete embeddings to calculate z")
        ("--kill_edit"; action=:store_true; help="dont use vae, z becomes zeros vector.")
        ("--temp"; arg_type=Float64; default=0.4; help="temperature sampler for mixsampler")
    end

    isa(args, AbstractString) && (args=split(args))

    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    options = parse_args(args, s)
    options["optim"] = eval(Meta.parse(options["optim"]))
    printConfig(options)
    config = condconfig = nothing
    if !isnothing(options["config"])
        config = eval(Meta.parse(read(options["config"], String)))
    end

    if !isnothing(options["condconfig"])
        condconfig = eval(Meta.parse(read(options["condconfig"], String)))
        condconfig["paug"] = options["paug"]
    end

    for (k,v) in options
        if haskey(config,k)
            config[k] = v
        else
            print(k,"\t")
        end
    end

    main(config,condconfig; generate=options["generate"] && !options["baseline"],
                        baseline=options["baseline"],
                        usegenerated=options["usegenerated"],
                        saveprefix=options["loadprefix"])
end

PROGRAM_FILE=="runexperiments_scan.jl" && run(ARGS)
#
# function get_experiments_rnn()
#     E          = [32, 64, 128]
#     B          = [4, 8, 16, 32]
#     lang       = ["spanish", "turkish"]
#     lr         = [0.001, 0.002]
#     exps = vec(collect(Iterators.product(E,B,lang,lr)))
# end
#
#
# #
# # function main_simple(config)
# #     println("Preprocessing Data & Initializing Model")
# #     processed, esets, model = get_data_model(config)
# #     println("Train Starts")
# #     train!(model, processed[1]; dev=processed[end])
# # end
#
# function run_experiments_rnn(baseconfig, id; fname="bathch_results.csv")
#     exps = get_experiments_rnn()
#     for (i,exp) in enumerate(exps)
#         if (i-1) % 8 == id
#             o = copy(baseconfig)
#             o["E"], o["B"], o["lang"], lr  = exp
#             o["optim"] = Adam(lr=lr)
#             main(o)
#         end
#     end
# end
#
# #run_experiments_rnn(rnnlm_sig_config, parse(Int,ARGS[1])-1)
#
# function get_experiments_recombine()
#     E          = [64, 128]
#     Z          = [8, 16]
#     B          = [4, 8, 16]
#     lang       = ["spanish", "turkish"]
#     lr         = [0.001, 0.002]
#     exps = vec(collect(Iterators.product(E,Z, B,lang,lr)))
# end
#
# function run_experiments_recombine(baseconfig, id; fname="bathch_results.csv")
#     exps = get_experiments_recombine()
#     for (i,exp) in enumerate(exps)
#         if (i-1) % 8 == id
#             o = copy(baseconfig)
#             o["E"], o["Z"], o["B"], o["lang"], lr  = exp
#             o["optim"] = Adam(lr=lr)
#             o["A"] = 2o["Z"]
#             o["copy"] = true
#             main(o)
#         end
#     end
# end
#
# #run_experiments_recombine(recombine_sig_config, parse(Int,ARGS[1])-1)
