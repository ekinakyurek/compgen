include("main.jl")

function get_experiments_rnn()
    E          = [32, 64, 128]
    B          = [4, 8, 16, 32]
    lang       = ["spanish", "turkish"]
    lr         = [0.001, 0.002]
    exps = vec(collect(Iterators.product(E,B,lang,lr)))
end




#
# function main_simple(config)
#     println("Preprocessing Data & Initializing Model")
#     processed, esets, model = get_data_model(config)
#     println("Train Starts")
#     train!(model, processed[1]; dev=processed[end])
# end

function run_experiments_rnn(baseconfig, id; fname="bathch_results.csv")
    exps = get_experiments_rnn()
    for (i,exp) in enumerate(exps)
        if (i-1) % 8 == id
            o = copy(baseconfig)
            o["E"], o["B"], o["lang"], lr  = exp
            o["optim"] = Adam(lr=lr)
            main(o)
        end
    end
end

#run_experiments_rnn(rnnlm_sig_config, parse(Int,ARGS[1])-1)

function get_experiments_recombine()
    E          = [64, 128]
    Z          = [8, 16]
    B          = [4, 8, 16]
    lang       = ["spanish", "turkish"]
    lr         = [0.001, 0.002]
    exps = vec(collect(Iterators.product(E,Z, B,lang,lr)))
end

function run_experiments_recombine(baseconfig, id; fname="bathch_results.csv")
    exps = get_experiments_recombine()
    for (i,exp) in enumerate(exps)
        if (i-1) % 8 == id
            o = copy(baseconfig)
            o["E"], o["Z"], o["B"], o["lang"], lr  = exp
            o["optim"] = Adam(lr=lr)
            o["A"] = 2o["Z"]
            o["copy"] = true
            main(o)
        end
    end
end
run_experiments_recombine(recombine_sig_config, parse(Int,ARGS[1])-1)
