using KnetLayers, Distributions
import KnetLayers: nllmask, findindices, _batchSizes2indices, IndexedDict, _pack_sequence
const parameters = KnetLayers.params
import Knet.SpecialFunctions: besseli,lgamma
const gpugc   = KnetLayers.gc
const arrtype = gpu()>=0 ? KnetArray{Float32} : Array{Float32}

###
#### UTILS
###
dir(path...) = joinpath(pwd(),path...)
const StrDict  = Dict{String,Int}
const VStrDict = Dict{String,Vector{Int}}
const CharDict = Dict{Char,Int}
CrossDict = Dict{Vector{Int},Vector{Int}}
###

function applymask(y, mask, f::Function)
    sy,sm = size(y),size(mask)
    @assert sy[2:end] == sm "input and mask size are different $sy and $sm"
    mask = reshape(mask,1,sm...)
    mask = convert(arrtype,mask)
    broadcast(f, y, mask)
end
applymask(y, mask::Nothing, f::Function) = y

function randchoice(v,num,L)
    l = length(v)
    if l==0
        Int[rand(1:L),rand(1:L)]
    elseif l < num
        rand(v,num)
    else
        v[randperm(l)[1:num]]
    end
end

function inserts_deletes(w1::Vector{T},w2::Vector{T}) where T
    D = T[]
    I = T[]
    for c1 in w1
        if c1 ∉ w2
            push!(D,c1)
        end
    end
    for c2 in w2
        if c2 ∉ w1
            push!(I,c2)
        end
    end
    return (I=I, D=D)
end

function tagstats(dict)
    ls = map(length,values(dict))
    xs = unique(ls)
    ys = map(z->length(findall(l->l==z,ls)), xs)
    return xs,ys
end

zeroarray(arrtype, d...; fill=0) = fill!(arrtype(undef,d...),fill)

function setoptim!(M, optimizer)
    for p in parameters(M)
        p.opt = deepcopy(optimizer);
    end
end

lrdecay!(M, decay::Real) =
    for p in parameters(M); p.opt.lr = p.opt.lr*decay; end

function transferto!(m1,m2)
    for (w1,w2) in zip(parameters(m1), parameters(m2))
        copyto!(value(w1), value(w2))
    end
end

function _batchSizes2ids(x)
    inds     = _batchSizes2indices(x.batchSizes) # converts cuda x.tokens, x.batchsizes -> vector of vectors format
    batchids = zeros(Int,length(x.tokens))
    for i=1:length(inds)
        batchids[inds[i]] .= i
    end
    return batchids
end

greedy(y) = mapslices(argmax, y, dims=1)

function trim(chars::Vector{Int},vocab)
    out = Int[]
    for c in chars
        c == specialIndicies.eow && break
        if c ∉ (specialIndicies.bow,specialIndicies.unk)
            push!(out,c)
        end
    end
    return join(vocab.tokens[out])
end

sample(y) = catsample(softmax(y;dims=1))

function catsample(p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end
end

catlast(x) = vcat(x[:,:,1],x[:,:,2])
drop(x) = dropout(x,0.4)

function PadRNNOutput2(y, indices)
    d      = size(y,1)
    B      = length(indices)
    lngths = length.(indices)
    Tmax   = maximum(lngths)
    cw     = Any[]
    @inbounds for i=1:B
        y1 = y[:,indices[i]]
        df = Tmax-lngths[i]
        if df > 0
            pad = fill!(arrtype(undef,d*df),0)
            ypad = reshape(cat1d(y1,pad),d,Tmax) # hcat(y1,kpad)
            push!(cw,ypad)
        else
            push!(cw,y1)
        end
    end
    reshape(vcat(cw...),d,B,Tmax)
end


function pad_packed_sequence(pseq, pad::Int; toend::Bool=true)
    if toend
        bs = _batchSizes2indices(pseq.batchSizes)
        _pack_sequence(map(s->[pseq.tokens[s];pad], bs))
    else
        bs = pseq.batchSizes
        B  = first(bs)
        (tokens = [fill!(Array{Int,1}(undef,B),pad);pseq.tokens], batchSizes=[B;bs])
    end
end

function nsample_packed_sequence(pseq, pad::Int; toend::Bool=true, nsample=500)
    inds = _batchSizes2indices(pseq.batchSizes)
    if toend
        inds =  map(s->[pseq.tokens[s];pad], inds)
    else
        inds =  map(s->[pad; pseq.tokens[s]], inds)
    end
    inds = repeat(inds, inner=nsample)
    x    = _pack_sequence(inds)
    x, _batchSizes2indices(x.batchSizes)
end

second(x) = x[2]
combinedicts(dict1::Dict, dict2::Dict) = combinedicts!(copy(dict1),dict2)
function combinedicts!(dict1::Dict, dict2::Dict)
    for (k,v) in dict2; dict1[k] = v; end
    return dict1
end

appenddicts(dict1::Dict, dict2::Dict) = appenddicts!(copy(dict1),dict2)
function appenddicts!(dict1::Dict, dict2::Dict)
    for (k,v) in dict2
        if !haskey(dict1,k)
            dict1[k] = length(dict1)+1
        end
    end
    return dict1
end

unzip(a) = map(x->getindex.(a, x), 1:length(first(a)))
function get_mask_sequence(lengths::Vector{Int}; makefalse=true)
    Tmax = maximum(lengths)
    if Tmax == 0
        if makefalse
            return falses(length(lengths), 1)
        else
            return trues(length(lengths),1)
        end
    else
        #Tmax == last(lengths) && return nothing
        B    = length(lengths)
        if makefalse
            mask = falses(length(lengths), Tmax)
        else
            mask = trues(length(lengths), Tmax)
        end
        for k=1:B
            @inbounds mask[k,lengths[k]+1:Tmax] .= (true & makefalse)
        end
        return mask
    end
end

function freebits(x; fb=4.0)
    mask = relu.(x .- fb)
    (x .* mask) ./ (mask .+ 1e-20)
end

struct Pw{T<:AbstractFloat}
    p::T; K::T; a::T; b::T; d::T; beta
end

function Pw{T}(p,K) where T
    dim = p-1
    a = (dim + 2K + sqrt(4K^2 + dim^2))/4
    b = (-2K  + sqrt(4K^2 + dim^2))/dim
    d = 4a*b/(1+b) - dim*log(dim)
    beta = Beta(dim/2, dim/2)
    return Pw(T(p),T(K),T(a),T(b),T(d),beta)
end

function sample(pw::Pw{T}) where T
    opb = (1+pw.b)
    omb = (1-pw.b)
    while true
        β  = rand(pw.beta)
        xp = (1-opb*β)
        xn = (1-omb*β)
        t  = 2*pw.a*pw.b / xn
        if (pw.p-1)*log(t) - t + pw.d - log(rand()) >= 0
            return T(xp / xn)
        end
    end
end

function sample_orthonormal_to(μ, p)
    v      = randn!(similar(μ))
    r      = sum(μ .* v,dims=1)
    ortho  = v .-  μ .* r
    ortho ./ sqrt.(sumabs2(ortho, dims=1))
end

add_norm_noise(μnorm, ϵ, max_norm) =
     min.(μnorm, max_norm-ϵ) .+ rand!(similar(μnorm)) .* ϵ

function sample_vMF(μ, ϵ, max_norm, pw=Pw{eltype(μ)}(size(μ,1),25); prior=false)
    if prior
        μ      = randn!(μ)
        μnorm  = sqrt.(sumabs2(μ, dims=1))
        zdir   = μ ./ μnorm
        znorm  = convert(arrtype,max_norm*rand(1,size(μ,2)))
        zdir .* znorm
    else
        w   = reshape([sample(pw) for i=1:size(μ,2)],1,size(μ,2))
        w   = convert(arrtype,w)
        μ   = μ .+ 1e-10
        μnorm    = sqrt.(sumabs2(μ, dims=1))
        μnoise   = add_norm_noise(μnorm, ϵ, max_norm)
        v        = sample_orthonormal_to(μ ./ μnorm, pw.p)
        scale    = sqrt.(1 .- w.^2)
        μscale   = μ .* w ./ μnorm
        (v .* scale  + μscale) .* μnoise
    end
end

function KnetLayers.PadSequenceArray(batch::Vector{Vector{T}}; pad=0) where T<:Integer
    B      = length(batch)
    lngths = length.(batch)
    Tmax   = maximum(lngths)
    padded = Array{T}(undef,B,Tmax)
    if Tmax == 0
        return pad * ones(T,B,1)
    else
        padded = Array{T}(undef,B,Tmax)
        @inbounds for n = 1:B
            padded[n,1:lngths[n]] = batch[n]
            padded[n,lngths[n]+1:end] .= pad
        end
        return padded
    end
end



# function expmask(y, mask)
#      s = size(y)
#      mask = reshape(mask,1,s[2:end]...)
#      y .- mask # mask should be Float32 array with masked locations are large
# end
#
# expmask(y, mask::Nothing) = y
#
# function normalmask(y, mask)
#     s = size(y)
#     mask = reshape(mask,1,s[2:end]...)
#     y .* arrtype(mask) # mask should be Float32 array with masked locations are large
# end
#
# normalmask(y, mask::Nothing) = y

# using Printf
# function prettymemory(b)
#     if b < 1000
#         value, units = b, "B"
#     elseif b < 1000^2
#         value, units = b / 1024, "KiB"
#     elseif b < 1000^3
#         value, units = b / 1024^2, "MiB"
#     else
#         value, units = b / 1024^3, "GiB"
#     end
#
#     if round(value) >= 100
#         str = string(@sprintf("%.0f", value), units)
#     elseif round(value * 10) >= 100
#         str = string(@sprintf("%.1f", value), units)
#     elseif value >= 0
#         str = string(@sprintf("%.2f", value), units)
#     else
#         str = "-"
#     end
#     return lpad(str, 7, " ")
# end
