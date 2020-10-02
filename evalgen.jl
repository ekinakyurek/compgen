# Direct Inference
include("exp.jl")
model  = KnetLayers.load(ARGS[1],"model")
config = model.config
task   = config["task"]
MT     = config["model"]
if haskey(config,"path") && task == SIGDataSet
    processed, esets, _, aug = read_from_jsons(defaultpath(config["task"]) * "/", config)
end
proc  = defaultpath(config["task"]) * "/" * config["splitmodifier"] * ".jld2"
println("processed file: ",proc," exist: ", isfile(proc))
processed, esets, vocab, embeddings = load_preprocessed_data(proc)
data = pickprotos_conditional(model, esets; subtask=nothing)
println("TEST EVAL")
println(condeval(model, data; k=3))
