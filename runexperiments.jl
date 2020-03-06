include("main.jl")

function run_experiments_simple(baseconfig, id; fname="bathch_results.csv")
    exps = get_experiments_simple()
    for (i,exp) in enumerate(exps)
        if i % 8 == id
            o = copy(baseconfig)
            o["H"], o["E"], o["Z"], lr , o["Kappa"], o["writedrop"], o["positions"], gclip  = exp
            o["A"] = 2o["Z"]
            o["optim"] = Adam(lr=lr, gclip=gclip)
            ppl = main_simple(o)
            open(fname,"a+") do f
                println(f,(exp..., ppl))
            end
        end
    end
end

run_experiments_simple(recombine_config, parse(Int,ARGS[1]))
