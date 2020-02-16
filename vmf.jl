using Distributions, Plots, Random
const arrtype = Array{Float64}

struct Pw{T<:AbstractFloat}
    p::T; K::T; b::T; x::T; c::T; beta
end

function Pw{T}(p,K) where T
    dim = p-1
    b   = dim / (sqrt(4K^2 + dim^2) + 2K)
    x   = (1 - b)/(1 + b)
    c   = K * x + dim * log(1 - x^2)
    beta = Beta(dim/2, dim/2)
    Pw(T(p),T(K),T(b),T(x),T(c),beta)
end

function sample(pw::Pw{T}) where T
    dim = pw.p-1
    while true
        z = rand(pw.beta)
        w = (1 - (1 + pw.b) * z) / (1 - (1 - pw.b) * z)
        if pw.K * w + dim * log(1 - pw.x * w) - pw.c >= log(rand()) #thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
            return w
        end
    end
end

l2norm(x; dims=1) = sqrt.(sum(xi->xi^2, x, dims=dims))

function sample_orthonormal_to(μ)
   v      = randn!(similar(μ))
   ortho  = v .-  μ .*  sum(μ .* v,dims=1)
   ortho ./ l2norm(ortho)
end

add_norm_noise(μnorm, ϵ, max_norm) =
     min.(μnorm, max_norm-ϵ) .+ rand!(similar(μnorm)) .* ϵ

function sample_vMF(μ, ϵ, max_norm, pw::Pw)
    w        = reshape([sample(pw) for i=1:size(μ,2)],1,size(μ,2))
    w        = convert(arrtype,w)
    μ        = μ .+ (1e-18 * randn!(similar(μ)))
    μnorm    = l2norm(μ)
    v        = sample_orthonormal_to(μ ./ μnorm) .*  sqrt.(1 .- w.^2)
    μscale   = μ .* w ./ μnorm
    (v + μscale) .* add_norm_noise(μnorm, ϵ, max_norm)
end

function main(kappa=15.0, eps=0.1)
    pw = Pw{Float64}(2,kappa)
    samples = sample_vMF(2ones(2,100), eps, 10, pw::Pw)
    x = samples[1,:]; y=samples[2,:]
    savefig(scatter(x,y; legend=false, markersize = 7.0,
    markeralpha = 1.0,
    markerstrokewidth = 0.0,
    markerstrokecolor = :black
    ),"julia_vmf.png")
end
main()
