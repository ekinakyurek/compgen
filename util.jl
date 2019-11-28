using KnetLayers
import KnetLayers: nllmask, arrtype, findindices, _batchSizes2indices, IndexedDict, _pack_sequence
const parameters = KnetLayers.params
const gpugc = KnetLayers.gc

function setoptim!(M, optimizer)
    for p in parameters(M)
        p.opt = deepcopy(optimizer);
    end
end

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
        c == vocab.specialIndices.eow && break
        if c ∉ vocab.specialIndices
            push!(out,c)
        end
    end
    return join(vocab.chars[out])
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

using Printf
function prettymemory(b)
    if b < 1000
        value, units = b, "B"
    elseif b < 1000^2
        value, units = b / 1024, "KiB"
    elseif b < 1000^3
        value, units = b / 1024^2, "MiB"
    else
        value, units = b / 1024^3, "GiB"
    end

    if round(value) >= 100
        str = string(@sprintf("%.0f", value), units)
    elseif round(value * 10) >= 100
        str = string(@sprintf("%.1f", value), units)
    elseif value >= 0
        str = string(@sprintf("%.2f", value), units)
    else
        str = "-"
    end
    return lpad(str, 7, " ")
end