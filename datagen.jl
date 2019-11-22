using KnetLayers, Random
import KnetLayers: IndexedDict, arrtype

setoptim!(M, optimizer) =
    for p in params(M); p.opt = deepcopy(optimizer); end

cross_vocabulary = join.(vec(collect(Iterators.product(0:9,'a':'z'))))
vocab            = IndexedDict([collect('0':'9'); collect('a':'z')])
digits           = '0':2:'8'
chars            = 'a':'h'
holdout          = ['0' .* ('a':'b'); '2'.*('c':'d'); '4'.*('e':'f'); '8'.*('g':'h'); "6h"]
data             = String[]
for d in digits
    for c in chars
        datum = string(d,c)
        datum ∉ holdout ? push!(data,datum) : continue
    end
end
x_onehot = [vocab[collect(d)] for d in data]
H = 256
V = length(vocab)

function encode(model, x_onehot, x)
    h  = model[1](x_onehot) # H x 2 x (T-1)
    H,N,T = size(h)
    h  = reshape(h, (H*N,T))
    e  = model[1](x) # H x N
    e  = reshape(e,(H*N,1))
    α1  = softmax(sum(e .* model[2](h), dims=1)) # 1 x T
    μ    = model[3](relu.(sum(α1 .* h, dims=2)))
    α2   = softmax(sum(e .* model[4](h), dims=1)) # 1 x T
    logσ² =  model[5](relu.(sum(α2 .* h, dims=2)))
    return μ, logσ²
end



function encode_watt(model, x_onehot, x)
#    h  = model[1](x_onehot) # H x 2 x (T-1)
#    H,N,T = size(h)
#    h  = reshape(h, (H*N,T))

    e  = model[1](x) # H x N
    H,N = size(e)
    e  = reshape(e,(H*N,1))
#    α1  = softmax(e .* model[2](h)) # 1 x T
    μ    = model[2](relu.(e))
#    α2   = softmax(e .* model[4](h)) # 1 x T
    logσ² =  model[3](relu.(e))
    return μ, logσ²
end



function encode(model, x_onehot, α1, α2)
    h  = model[1](x_onehot) # H x 2 x (T-1)
    H,N,T = size(h)
    h  = reshape(h, (H*N,T))
    μ    = model[3](relu.(sum(α1 .* h, dims=2)))
    logσ² =  model[5](relu.(sum(α2 .* h, dims=2)))
    return μ, logσ²
end


function decode(model,z)
    return model[end](z)
end


function decode_watt(model, x_onehot, z)
    h  = model[1](x_onehot) # H x 2 x (T-1)
    H,N,T = size(h)
    h  = reshape(h, (H*N,T))
    α1  = softmax(sum(z .* model[end-2](h),dims=1)) # 1 x T
    μ    = model[end-1](relu.(sum(α1 .* h, dims=2)))
    return model[end](reshape(μ,H,N))
end


function loss(model, x_onehot, x)
    μ, logσ² =  encode(model, x_onehot,x)
    σ² = exp.(logσ² )
    σ  = sqrt.(σ²)
    z  = μ .+ randn!(similar(μ)) .* σ
    H  = size(z,1) ÷ 2
    xp = decode(model, reshape(z,H,2))
    KL =  -sum(@. 1 + logσ² - μ*μ - σ²) / 2 / (length(xp))
    return nll(xp,x) + KL
end

function loss_watt(model, x_onehot, x)
    μ, logσ² =  encode_watt(model, x_onehot,x)
    σ² = exp.(logσ² )
    σ  = sqrt.(σ²)
    z  = μ .+ randn!(similar(μ)) .* σ
    H  = size(z,1) ÷ 2
    xp = decode_watt(model,x_onehot, z)
    KL =  -sum(@. 1 + logσ² - μ*μ - σ²) / 2 / (length(xp))
    return nll(xp,x) + KL
end


function c_sample(model, vocab, x_onehot; N=10)
    μ, logσ² =  avg_dist_parameters(model,x_onehot)
    fill!(μ,0)
    fill!(logσ²,0)
    return values = [sample(model,vocab,μ,logσ²) for i=1:N]
end

function c_sample_by_α(model,vocab,x_onehot,α; N=10)
    α = Float32.(α) / sum(α)
    xt = hcat(x_onehot...)
    μ, σ² =  encode(model, xt, α, α)
    values = []
    for i=1:N
        v = sample(model,vocab,μ,σ²)
        push!(values,v)
    end
    return values
end


function sample(model, vocab, μ, logσ²)
    σ  = sqrt.(exp.(logσ²))
    z  = μ .+ randn!(similar(μ)) .* σ
    xp = model[end](reshape(z,H,2))
    vocab[mapslices(argmax,xp,dims=1)]
end

function avg_dist_parameters(model,x_onehot)
    μ, σ² = nothing,nothing
    for i=1:length(x_onehot)
        xt = hcat([x_onehot[1:i-1]; x_onehot[i+1:end]]...)
        x  = x_onehot[i]
        μi, logσ² =  encode_watt(model, xt, x)
        if μ == nothing
            μ = μi; σ² = exp.(logσ²)
        else
            μ += μi; σ² += exp.(logσ²);
        end
    end
    return μ/length(x_onehot),  σ²/length(x_onehot)
end


function train!(model, x_onehot; epoch=20, optim=Adam())
    setoptim!(model,optim)
    for i=1:epoch
        lss = 0.0
        cnt = 0
        for i=1:length(x_onehot)
            xt = hcat([ x_onehot[1:i-1]; x_onehot[i+1:end] ]...)
            x  = x_onehot[i]
            J = @diff loss(model, xt, x)
            lss += value(J)
            cnt += 1
            for w in params(J)
                update!(value(w), grad(J,w), w.opt)
            end
        end
    end
end

encoder = Embed(input=V,output=H)
decoder = Embed(input=H,output=V)
Wμa     = Multiply(input=2H,output=2H)
Wμ      = Linear(input=2H,output=2H)
Wσa     = Multiply(input=2H,output=2H)
Wσ      = Linear(input=2H,output=2H)
model   = [encoder,Wμa,Wμ,Wσa,Wσ, decoder]
model2  = [encoder,Wμ, Wσ, Wμa,Wμa, decoder]

for opt in (Adam, Rmsprop, SGD)
    for lr in (0.0001, 0.001, 0.01, 0.1)
        for epoch in (10, 20, 30)
            cmodel = deepcopy(model)
            train!(cmodel, x_onehot;epoch=epoch, optim=opt(lr=lr))
            sampled = c_sample(cmodel, vocab, x_onehot; N=100)
            unseen = [join(s) for s in sampled if join(s) ∈ holdout]
            println("stats for $opt(lr=$lr), epoch: $epoch ->", length(unseen)," , ", unique(unseen))
        end
    end
end
