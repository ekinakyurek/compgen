using ArgParse
include("exp.jl")

function run(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler=ArgParse.debug_handler
    s.description = "SIGDataSet Experiments"
    @add_arg_table! s begin
        ("--seed"; arg_type=Int; default=0; help="seed")
        ("--hints"; arg_type=Int; default=4; help="test set hints in training set")
        ("--config"; arg_type=String; help="generative model's config file")
        ("--condconfig"; arg_type=String; help="conditional model's config file")
        ("--H"; arg_type=Int; default=1024; help="hidden dim")
        ("--E"; arg_type=Int; default=1024; help="embedding dim")
        ("--Z"; arg_type=Int; default=1; help="embedding dim")
        ("--nproto"; arg_type=Int; default=2; help="prototype count")
        ("--Nlayers"; arg_type=Int; default=1; help="number of rnn layers")
        ("--B"; arg_type=Int; default=64; help="batch size")
        ("--epoch"; arg_type=Int; default=100; help="epoch")
        ("--attdim"; arg_type=Int; default=1024; help="attention dim")
        ("--gradnorm"; arg_type=Float64; default=0.0; help="global gradient norm clip, 0 for none")
        ("--writedrop"; arg_type=Float64; default=0.0; help="write dropout in copy")
        ("--outdrop"; arg_type=Float64; default=0.0; help="output dropout in rnn")
        ("--pdrop"; arg_type=Float64; default=0.0; help="dropout in various locations")
        ("--attdrop"; arg_type=Float64; default=0.0; help="attention dropout in attentions ")
        ("--temp"; arg_type=Float64; default=0.05; help="temperature sampling param")
        ("--optim"; arg_type=String; default="Adam(;lr=0.0001)")
        ("--lang"; arg_type=String; default="turkish")
        ("--subtask"; arg_type=String; default="reinflection"; help="subtask for generative model, analyses or reinflection")
        ("--copy"; action=:store_true; help="copy meachanism in rnn")
        ("--outdrop_test"; action=:store_true; help="dropout on output at test time")
        ("--seperate"; action=:store_false; help="seperate encodings for 2 prototype")
        ("--seperate_emb"; action=:store_true; help="seperate embeddings for input-output")
        ("--kill_edit"; action=:store_true; help="no vae")
        ("--paug"; arg_type=Float64; default=.0; help="augmentation ratio for condtional model, 0 for direct concatenation")
        ("--baseline"; action=:store_true; help="run conditional model without augmentation")
        ("--usegenerated"; action=:store_true; help="run conditional model without augmentation")
        ("--generate"; action=:store_true; help="train generative model")
        ("--N"; arg_type=Int; default=180; help="number of rnn layers")
        ("--Nsamples"; arg_type=Int; default=180; help="number of rnn layers")
        ("--rare_token"; action=:store_true; help="filter rare tokens")
        ("--loadprefix"; help="sample file prefix")
        ("--modeldir"; help="model dir")
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
        condconfig["lang"] = options["lang"]
    end

    for (k,v) in options
        if haskey(config,k)
            config[k] = v
        else
            print(k,"\t")
        end
    end
    printConfig(config)
    main(config,condconfig; generate=options["generate"] && !options["baseline"],
                            baseline=options["baseline"],
                            usegenerated=options["usegenerated"],
                            saveprefix=options["loadprefix"])
end

PROGRAM_FILE=="runexperiments.jl" && run(ARGS)
