include("main.jl")
using ArgParse
id = parse(Int,ARGS[1])

function printstats(result)
    samples_in_trn,samples_in_test,samples_in_dev = map(length,result.existsamples)
    mi = result.mi
    testppl = result.testppl
    dictppl = result.dictppl
    return "$samples_in_trn,$samples_in_test,$samples_in_dev,$mi,$testppl,$dictppl"
end


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "id"
        help = "gpu id"
        arg_type=Int
        default= 0
        "--lang"
        help = "turkish|spanish"
        arg_type = String
        default = "turkish"
        "--Z", "--E"
        help = "Z is latent dimension, E is embedding dimension"
        arg_type = Int
        default = 8    
        "--H"
        help = "H is hidden dimension of the LSTMs"
        arg_type = Int
        default = 512
        "--B"
        help = "B is batchsize"
        arg_type = Int
        default = 16
        "--kl_rate"
        help = "increase steps kl weight"
        arg_type = Float64
        default = 0.1
        "--fb_rate"    
        help = "Free Bits rate"
        arg_type = Float64
        default = 4.0
        "--lr"
        help = "Learning Rate"
        arg_type = Float64
        default = 0.002
    end
    parse_args(s)
end

const config_keys = ("lang","B","H","E","Z","aepoch","epoch","lr","kl_rate","fb_rate","pdrop","concatz")
function printconfig(o)
    str = ""
    for k in config_keys
        str *= o[k]
        str *= ','
    end
    str
end

function runsinglegpu()
    o = parse_commandline()
    result = main(:VAE;lang=o["lang"],
                  B=o["B"], E=o["E"], Z=o["Z"], H=o["H"],
                  epoch=o["epoch"],
                  optim=Adam(lr=o["lr"]),
                  fb_rate=o["fb_rate"] |> Float32,
                  kl_rate=o["kl_rate"] |> Float32,
                  pdrop=o["pdrop"],
             	  aepoch=o["aepoch"],
                  concatz=o["concatz"])

    try 
        open("results.csv","a+") do f
            println(f,"$(printconfig(o))$(printstats(result))")
        end
    catch
        println("couldn't add below line to result file: ")
        println("$(printconfig(o))$(printstats(result))")
    end
end

runsinglegpu()

# function runexperiments(id)
#     i  = 0
#     experiments = []
#     for lang in ("spanish", "turkish"),
#         B in (16),
#         H in (512),
#         epoch in (1),
#         concatz in (true),
#         E in (8, 16, 32),
#         Z in (8, 16, 32), 
#         aepoch in (1), #(10, 20, 30),
#         kl_rate in (0.1f0, 0.05f0),
#         pdrop in (0.0, 0.4),
#         fb_rate in (2.0f0, 4.0f0, 8.0f0),
#         lr in (0.001, 0.002, 0.004)
#         o = (lang=lang,B=B,H=H,E=E,Z=Z,
#              epoch=epoch,concatz=concatz,
#              aepoch=aepoch,kl_rate=kl_rate,
#              pdrop=pdrop,fb_rate=fb_rate,lr=lr) 
#         push!(experiments,o)
#     end

#     for (i,o) in enumerate(experiments)
#         GC.gc(); gpugc()
#         if (i % 8)+1 == id
#            result = main(:VAE;lang=o.lang, 
#                          B=o.B, E=o.E, Z=o.Z, H=o.H,
#                          epoch=o.epoch, 
#                          optim=Adam(lr=o.lr), 
#                          fb_rate=o.fb_rate, 
#                          kl_rate=o.kl_rate,
#                          pdrop=o.pdrop, 
#                          aepoch=o.aepoch,
#                          concatz=o.concatz)
            
#             try 
#                 open("results.csv","a+") do f
#                     println(f,"$o.B,$o.H,$o.E,$o.Z,$o.aepoch,$o.epoch,$o.kl_rate,$o.pdrop,$o.fb_rate,$o.lr,$o.concatz,$(printstats(result))")
#                 end
#             catch
#                 println("couldn't add below line to result file: ")
#                 println("$o.B,$o.H,$o.E,$o.Z,$o.aepoch,$o.epoch,$o.kl_rate,$o.pdrop,$o.fb_rate,$o.lr,$o.concatz,$(printstats(result))")
#             end
            
#         end
#         i+=1
#     end
# end



#runexperiments(id+1)
