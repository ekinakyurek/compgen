using LibGit2, Random, KnetLayers
import KnetLayers: IndexedDict, _pack_sequence

"""
    const specialTokens = (unk="❓", mask="⭕️", eow="🏁", bow="🎬")
    unk: unknown inputs
    mask: mask token
    eow: end of word/sentence
    bow: beginning of word/sentence
"""
const specialTokens = (unk="❓", mask="⭕️", eow="🏁", bow="🎬")

struct Vocabulary
    chars::IndexedDict{Char}
    tags::IndexedDict{String}
    specialTokens::NamedTuple
    specialIndices::NamedTuple
end

"""
    DataSet
    Abstract type for datasets
"""
abstract type DataSet; end

"""
    SIGDataSet
    Sigmorphon Data Set
"""
abstract type SIGDataSet <: DataSet; end

const StrDict  = Dict{String,Int}
const VStrDict = Dict{String,Vector{Int}}
const CharDict = Dict{Char,Int}

"""
     parseDataLine(line::AbstractString, p::Parser{<:MyDataSet}; wLemma=true, parseAll)
     It parses a line from given dataset. Returns a ParsedIO object.
"""
function parseDataLine(line::AbstractString)
    lemma, surface, tokens = split(line, '\t')
    (surface=collect(surface), lemma=collect(lemma), tags=split(tokens,';'))
end

dir(path...) = joinpath(pwd(),path...)
download(dataset::Type{SIGDataSet}; path=dir("data","Sigmorphon")) = !isdir(path) ? LibGit2.clone("https://github.com/sigmorphon/conll2018", path) : true
    
"""
    EncodedFormat
    EncodedFormat is structure that keeps morphological anlysis of a word with encoded fields.
"""
struct EncodedFormat
    surface::Vector{Int}
    lemma::Vector{Int}
    tags::Vector{Int}
end

EncodedFormat(a::NamedTuple, v::Vocabulary) =
EncodedFormat(map(x->get(v.chars, x, v.specialIndices.unk)::Int, a.surface),
              map(x->get(v.chars, x, v.specialIndices.unk)::Int, a.lemma),
              map(x->get(v.tags, x, v.specialIndices.unk)::Int, a.tags))


encode(s::NamedTuple, v::Vocabulary) = EncodedFormat(s,v)

CrossDict = Dict{Vector{Int},Vector{Int}}

function encode(data::Vector, v::Vocabulary; num=2) 
    lemma2loc, morph2loc = CrossDict(), CrossDict()
    edata = Array{EncodedFormat,1}(undef,length(data))
    for (i,datum) in enumerate(data)
          encoded = edata[i] = encode(datum,v)
          push!(get!(lemma2loc, encoded.lemma, Int[]),i)
          push!(get!(morph2loc, encoded.tags, Int[]), i)     
    end
    MorphData(edata,lemma2loc,morph2loc,num)
end

function Vocabulary(data::Vector)
    char2ix, tag2ix, lemma2loc, morph2loc = CharDict(), StrDict(), VStrDict(), VStrDict()

    for (i,T) in enumerate(specialTokens)
         get!(char2ix,T[1],i); get!(tag2ix,T,i);
    end
    
    specialIndicies = (unk=1, mask=2, eow=3, bow=4)

    for (i,(surface, lemma, tags)) in enumerate(data)
        
        for c::Char in surface
            get!(char2ix, c, length(char2ix)+1)
        end
        
        for c::Char in lemma
            get!(char2ix, c, length(char2ix)+1)
        end
        
        for t::String in tags
            get!(tag2ix,t,length(tag2ix)+1)
        end
    end
        
    Vocabulary(IndexedDict(char2ix), IndexedDict(tag2ix),  specialTokens, specialIndicies)
end

function tagstats(dict)
    ls = map(length,values(dict))
    xs = unique(ls)
    ys = map(z->length(findall(l->l==z,ls)), xs)
    return xs,ys
end


struct MorphData{T <: AbstractVector}
    data::T
    lemma2loc::CrossDict
    morph2loc::CrossDict
    num::Int
end

Base.length(m::MorphData) = length(m.data)

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

function Base.iterate(m::MorphData, s...)
    next = iterate(m.data, s...)
    next === nothing && return nothing
    (v, snext) = next
    index = snext-1
    L = length(m)
    surface, lemma, tags = v.surface, v.lemma, v.tags
    ex1 = [sfs for sfs in m.lemma2loc[lemma] if sfs != index]
    ex2 = [sfs for sfs in m.morph2loc[tags]  if sfs != index]
    s1 =  map(i->m.data[i].surface, randchoice(ex1, m.num, L))
    s2 =  map(i->m.data[i].surface, randchoice(ex2, m.num, L)) 
    others =  map(i->m.data[i].surface, rand(1:L,4))                   
    sc =  [[surface];s1;s2;others]
    length(sc) == 0 && error()
    r = sortperm(sc,by=length,rev=true)                       
    ex =  sc[r] 
    return (surface, ex, sortperm(r)), snext
end


function getbatch(iter,B)
    edata = collect(Iterators.take(iter,B))
    if (b = length(edata)) != 0
        sfs,exs, perms = first.(edata), second.(edata), last.(edata)
        r   = sortperm(sfs, by=length, rev=true)
        xi = _pack_sequence(sfs[r])
        xt = map(_pack_sequence,exs[r])
        return xi, xt, perms[r]
    else
        return nothing
    end
end

