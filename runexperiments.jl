include("main.jl")
id = parse(Int,ARGS[1])

function printstats(result)
    samples_in_trn,samples_in_test,samples_in_dev = map(length,result.existsamples)
    mi = result.mi
    testppl = result.testppl
    dictppl = result.dictppl
    return "$samples_in_trn,$samples_in_test,$samples_in_dev,$mi,$testppl,$dictppl"
end

function runexperiments(id)
    i  = 0
    for lang in ("spanish", "turkish"),
        B in (16),
        H in (512),
        epoch in (30),
        concatz in (true),
        E in (8, 16, 32),
        Z in (8, 16, 32), 
        aepoch in (10, 20, 30),
        kl_rate in (0.1, 0.05),
        pdrop in (0.0, 0.4),
        fb_rate in (2.0, 4.0, 8.0),
        lr in (0.001, 0.002, 0.004)
        if (i % 8)+1 == id
           result = main(VAE;lang=lang, 
                       B=B, E=E, Z=Z, H=H,
                       epoch=epoch, 
                       optim=Adam(lr=lr), 
                       fb_rate=fb_rate, 
                       kl_rate=kl_rate,
                       pdrop=pdrop, 
                       aepoch=aepoch,
                       concatz=concatz)
            try 
                open("results.csv","a+") do f
                    println(f,"$B,$H,$E,$Z,$aepoch,$epoch,$kl_rate,$pdrop,$fb_rate,$lr,$concatz,$(printstats(result))")
                end
            catch
                println("couldn't add below line to result file: ")
                println("$B,$H,$E,$Z,$aepoch,$epoch,$kl_rate,$pdrop,$fb_rate,$lr,$concatz,$(printstats(result))")
            end
            gpugc()
        end
        i+=1
    end
end


runexperiments(id+1)
