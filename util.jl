using Knet
gpu()<0 && gpu(0)
using KnetLayers, Distributions
import KnetLayers: Knet, nllmask, findindices, _batchSizes2indices, IndexedDict, _pack_sequence
const parameters = KnetLayers.params
import Knet.SpecialFunctions: besseli,loggamma
using ClearStacktrace
const gpugc   = KnetLayers.gc
const arrtype = gpu()>=0 ? KnetArray{Float32} : Array{Float32}
@eval Knet @primitive bmm(x1,x2; transA::Bool=false, transB::Bool=false),dy,y (transA ? bmm(x2, dy; transA=transB , transB=true) :  bmm(dy, x2;  transA=false, transB=!transB) )    (transB ? Knet.bmm(dy,x1; transA=true , transB=transA) :  bmm(x1, dy;  transA=!transA , transB=false))
using Printf, Dates, JSON
###
#### UTILS
###
dir(path...) = joinpath(pwd(),path...)
const StrDict  = Dict{String,Int}
const VStrDict = Dict{String,Vector{Int}}
const CharDict = Dict{Char,Int}
CrossDict = Dict{Vector{Int},Vector{Int}}
###

function expand(x; dim=1)
    shape,ND = size(x), ndims(x)
    if dim <= ND + 1
        reshape(x, (shape[1:dim-1]...,1, shape[dim:end]...))
    else
        reshape(x, (shape...,ntuple(i->1, dim-ND)))
    end
end
function applymask(y, mask, f::Function)
    sy,sm = size(y),size(mask)
    @assert sy[2:end] == sm "input and mask size are different $sy and $sm"
    mask = reshape(mask,1,sm...)
    mask = convert(arrtype,mask)
    broadcast(f, y, mask)
end
function applymask2(y, mask, f::Function)
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

function inserts_deletes(input::Vector{T},proto::Vector{T}) where T
    inp   = Set(input)
    pro   = Set(proto)
    I     = collect(setdiff(inp,pro))
    D     = collect(setdiff(pro,inp))
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
        if isnothing(p.opt)
            p.opt = deepcopy(optimizer)
        end
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
    join(vocab.tokens[trimencoded(chars)],' ')
    # out = Int[]
    # for c in chars
    #     c == specialIndicies.eow && break
    #     if c ∉ (specialIndicies.bow,specialIndicies.unk)
    #         push!(out,c)
    #     end
    # end
    # return join(vocab.tokens[out],' ')
end

sample(y) = catsample(softmax(y;dims=1))

function catsample(p)
    p = p ./ sum(p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end
end

function finalstates(hidden)
    if ndims(hidden)==3 && size(hidden,3) > 1
        vcat(hidden[:,:,end-1],hidden[:,:,end])
    else
        return reshape(hidden, size(hidden,1), size(hidden,2))
    end
 end

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

function freebits(x, fb=4.0)
    mask = relu.(x .- fb)
    (x .* mask) ./ (mask .+ 1e-20)
end

# struct Pw{T<:AbstractFloat}
#     p::T; K::T; a::T; b::T; d::T; beta
# end

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
    # a = (dim + 2K + sqrt(4K^2 + dim^2))/4
    # b = (-2K  + sqrt(4K^2 + dim^2))/dim
    # d = 4a*b/(1+b) - dim*log(dim)
    # beta = Beta(dim/2, dim/2)
    # return Pw(T(p),T(K),T(a),T(b),T(d),beta)
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
    # opb = (1+pw.b)
    # omb = (1-pw.b)
    # while true
    #     β  = rand(pw.beta)
    #     xp = (1-opb*β)
    #     xn = (1-omb*β)
    #     t  = 2*pw.a*pw.b / xn
    #     if (pw.p-1)*log(t) - t + pw.d - log(rand()) >= 0
    #         return T(xp / xn)
    #     end
    # end
#end

l2norm(x; dims=1) = sqrt.(sumabs2(x, dims=dims))

function sample_orthonormal_to(μ)
   v      = randn!(similar(μ))
   ortho  = v .-  μ .*  sum(μ .* v,dims=1)
   ortho ./ l2norm(ortho)
end

add_norm_noise(μnorm, ϵ, max_norm) =
     min.(μnorm, max_norm-ϵ) .+ rand!(similar(μnorm)) .* ϵ

function sample_vMF(μ, ϵ, max_norm, pw::Pw; prior=false)
    if prior
        μ      = randn!(μ)
        μnorm  = l2norm(μ)
        zdir   = μ ./ μnorm
        znorm  = convert(arrtype,max_norm*rand(1,size(μ,2)))
        zdir .* znorm
    else
        w        = reshape([sample(pw) for i=1:size(μ,2)],1,size(μ,2))
        w        = convert(arrtype,w)
        μ        = μ .+ (Float32(1e-18) * randn!(similar(μ)))
        μnorm    = l2norm(μ)
        v        = sample_orthonormal_to(μ ./ μnorm) .*  sqrt.(1 .- w.^2)
        μscale   = μ .* w ./ μnorm
        (v + μscale) .* add_norm_noise(μnorm, ϵ, max_norm)
    end
end

# function KnetLayers.PadSequenceArray(batch::Vector{Vector{T}}; pad=0, makefalse=true) where T<:Integer
#     B      = length(batch)
#     lngths = length.(batch)
#     Tmax   = maximum(lngths)
#     padded = Array{T}(undef,B,Tmax)
#     if Tmax == 0
#         return (tokens=pad * ones(T,B,1), mask=makefalse .& trues(B,1))
#     else
#         padded = Array{T}(undef,B,Tmax)
#         @inbounds for n = 1:B
#             padded[n,1:lngths[n]] = batch[n]
#             padded[n,lngths[n]+1:end] .= pad
#         end
#
#         return (tokens=padded, mask=!makefalse .⊻ (padded .== pad))
#     end
# end

function KnetLayers.PadSequenceArray(batch::Vector{Vector{T}}; pad=0, makefalse=true) where T<:Integer
    lngths = map(length,batch)
    Tmax   = maximum(lngths)
    B = length(batch)
    if Tmax == 0
        (tokens=pad * ones(T,B,1), mask=makefalse .& trues(B,1))
    else
        padded = fill!(Array{T}(undef,length(batch),Tmax),pad)
        for (n,l) in enumerate(lngths)
            @inbounds padded[n,1:l] = batch[n]
        end
        (tokens=padded, mask=!makefalse .⊻ ( (padded .== pad) .| (padded .== specialIndicies.unk)))
    end
end



function sentenceEmb(sentence, wordVectors, dim)
    words = split(sentence)
    wordEmbs = initEmbRandom(dim, length(words))
    for (idx, word) in enumerate(words)
        if haskey(wordVectors,word)
            wordEmbs[:,idx] .= wordVectors[word]
        end
    end
    return vec(sum(wordEmbs,dims=2))
end

function splitdata(data::Vector, r::Vector{<:Real})
    ls = isa(r[1],AbstractFloat) ? floor.(Int, r .* length(data)) : r #Ratio or Number
    map(rng->data[rng[1]:rng[2]], zip(cumsum(ls) .- ls .+ 1, cumsum(ls)))
end

function initializeWordEmbeddings(dim; wordsDict = nothing, prefix = "data/Glove/glove.6B.")
    # default dictionary to use for embeddings
    wordVectors  = Dict()
    open(prefix*string(dim)*"d.txt") do f
        for line in eachline(f)
            line = split(strip(line))
            word = lowercase(line[1])
            vector = [parse(Float32,x) for x in line[2:end]]
            wordVectors[word] = vector
        end
    end
    σ2, μ = std_mean(collect(values(wordVectors)))
    embeddings = randn(dim,length(wordsDict)) .* sqrt.(σ2) .+ μ
    for (w,index) in wordsDict
        if occursin(" ",w)
            embeddings[:,index] .= sentenceEmb(w, wordVectors)
        else
            if haskey(wordVectors,w)
                embeddings[:,index] .= wordVectors[w]
            end
        end
    end
    return embeddings
end

load_embed(vocab, config, embeddings::Nothing) =
    Embed(input=length(vocab),output=config["E"])

load_embed(vocab, config, embeddings::AbstractArray) =
    Embed(param(convert(arrtype,copy(embeddings))))

expand_hidden(h,B) = expand(h,dim=2) .+ zeroarray(arrtype,size(h,1),B,size(h,2))
batch_last(x) = permutedims(x, (1,3,2))

function _repeat(y, cnt::Int; dim=1)
    h = expand(y,dim=dim)
    s = ntuple(i->(i==dim ? cnt : 1),ndims(h))
    h .+ zeroarray(arrtype,s...)
end
function negativemask!(y,inds...)
    T = eltype(y)
    for i in inds
        y[i,:] .= -T(1.0f18)
    end
end

function trimencoded(x)
    stop = findfirst(i->i==specialIndicies.eow,x)
    stop = isnothing(stop) ? length(x) : stop
    return x[1:stop-1]
end

import KnetLayers: load, save
function save_preprocessed_data(m, data, esets, embeddings)
    fname = prefix(m.config["task"], m.config) * "_processesed.jld2"
    #save(fname, "data", data, "esets", esets, "tokens", m.vocab.tokens, "inpdict", m.vocab.inpdict, "outdict", m.vocab.outdict, "embeddings", embeddings)
    save(fname, "data", data, "esets", esets, "vocab", m.vocab, "embeddings", embeddings)
end

function load_preprocessed_data(config)
    task = config["task"]
    d = load(prefix(task, config) * "_processesed.jld2")
    #p = Parser{task}()
    get(d,"data",d["esets"]), d["esets"], d["vocab"], get(d,"embeddings",nothing) #Vocabulary(d["tokens"], d["inpdict"], d["outdict"],p), get(d,"embeddings",nothing)
end

function Base.sortperm(A::AbstractMatrix; dims::Integer, rev::Bool = false)
    P = mapslices(x -> sortperm(x; rev = rev), A, dims = dims)
    if dims == 1
        for j = 1:size(P, 2)
            offset = (j - 1) * size(P, 1)
            for i = 1:size(P, 1)
                P[i, j] += offset
            end
        end
    else # if dims == 2
        for j = 1:size(P, 2)
            for i = 1:size(P, 1)
                P[i, j] = (P[i, j] - 1) * size(P, 1) + i
            end
        end
    end
    return P
end

import Knet: _ser, JLDMODE, save

function save(f::AbstractString, nt::NamedTuple)
    KnetLayers.save(f,Dict(zip(string.(keys(nt)), nt)))
end

function clip_gradient_norm_(J, max_norm=Inf; p=2)
    weights  = parameters(J)
    if isinf(p)
        total_norm = maximum((abs.(grad(J,w)) for w in weights))
    else
        total_norm = 0.0
        for w in weights
            total_norm +=  norm(grad(J,w)) .^ 2
        end
        total_norm = sqrt(total_norm)
    end
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1
        for w in weights
            Knet.lmul!(clip_coef, grad(J,w))
        end
    end
    return total_norm
end




# function attend2(x, projX, h, projH, att, mask=nothing; sumout=true, pdrop=0.4)
#     α, interscore = interact2(projX(x), projH; sumout=sumout, mask=mask) # 1,B,T'
#     y   = att(mat(sum(α .* h , dims=3), dims=1))
#     y, α, interscore
# end
#
# function interact2(x, h; sumout=true, mask=nothing)
#     y    = mat(applymask(sum(x .* h,dims=1),mask,-)) # B,T'
#     α    = reshape(softmax(y,dims=2),1,size(y)...)   # 1,B,T'
#     return α,y
# end




# function getcindex(V,xp,pred)
#     if pred > V
#         return xp[pred-V]
#     end
#     return pred
# end


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
