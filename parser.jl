using LibGit2, Random, KnetLayers, StringDistances
import KnetLayers: IndexedDict, _pack_sequence
abstract type AbstractVAE{T} end

const specialTokens   = (unk="❓", mask="⭕️", eow="🏁", bow="🎬", sep="→")
const specialIndicies = (unk=1, mask=2, eow=3, bow=4, sep=5)

"""
    DataSet
    Abstract type for datasets
"""
abstract type DataSet; end
abstract type SIGDataSet <: DataSet; end
abstract type SCANDataSet <: DataSet; end
abstract type YelpDataSet <: DataSet; end

struct Parser{MyDataSet}
    regex::Union{Regex,Nothing}
    partsSeperator::Union{Char,Nothing}
    tagsSeperator::Union{Char,Nothing}
    freewords::Union{Set{String},Nothing}
end

Parser{SIGDataSet}(version=:default)  = Parser{SIGDataSet}(nothing,'\t',';', nothing)
Parser{SCANDataSet}(version=:default) = Parser{SCANDataSet}(r"^IN\:\s(.*?)\sOUT\: (.*?)$",' ',nothing, nothing)
Parser{YelpDataSet}(version=:default) = Parser{YelpDataSet}(nothing,'\t',nothing,
                                                            Set(readlines(defaultpath(YelpDataSet)*"/free.txt")))

defaultpath(dataset::Type{SIGDataSet}) = dir("data","Sigmorphon")
defaultpath(dataset::Type{SCANDataSet}) = dir("data","SCAN")
defaultpath(dataset::Type{YelpDataSet}) = dir("data","Yelp")

function download(dataset::Type{SIGDataSet}; path=defaultpath(SIGDataSet)) 
    !isdir(path) && LibGit2.clone("https://github.com/sigmorphon/conll2018", path)
    unipath = "data/unimorph"
    if !isdir(unipath)
        mkdir(unipath)
        server = "https://github.com/unimorph/"
        for lang in ("eng", "spa", "tur")
            langpath = joinpath(unipath,lang)
            !isdir(langpath) && LibGit2.clone(server*lang, joinpath(langpath))
        end
    end
    return true
end
download(dataset::Type{SCANDataSet}; path=defaultpath(SCANDataSet)) =
    !isdir(path) ? LibGit2.clone("https://github.com/brendenlake/SCAN", path) : true

function download(dataset::Type{YelpDataSet}; path=defaultpath(YelpDataSet))
    if !isdir(data/Glove)
        download("http://nlp.stanford.edu/data/glove.6B.zip","data/glove.6B.zip")
        run(`unzip -qq data/glove.6B.zip -d data/Glove/`)
        rm("data/glove.6B.zip")
    else
        @warn "There is already data/Glove folder, skipping..."
    end
    if !isdir("data/Yelp")
        mkdir("data/Yelp")
        server = "https://worksheets.codalab.org/rest/bundles/0x984fe19b60f1479b925933eacbfda8d8/contents/blob/"
        for file in ["train","test","valid"] .*  ".tsv"
            Base.download(joinpath(server,file), joinpath(path,file))
        end
        Base.download(joinpath(server,"free.txt"),  joinpath(path,"free.txt"))
    else
        @warn "There is already data/Yelp folder, skipping..."
    end
    return true
end

prefix(dataset::Type{SCANDataSet}, opt) =
    joinpath(defaultpath(SCANDataSet), string(opt["model"],"_",opt["task"],"_",opt["split"],"_condition_",opt["conditional"]))
prefix(dataset::Type{SIGDataSet}, opt) =
    joinpath(defaultpath(SIGDataSet), string(opt["model"],"_",opt["task"],"_",opt["lang"],"_condition_",opt["conditional"]))
prefix(dataset::Type{YelpDataSet}, opt) =
    joinpath(defaultpath(YelpDataSet), string(opt["model"], "_", opt["task"]))


function rawfiles(dataset::Type{YelpDataSet}, config)
    ["data/Yelp/train.tsv",
     "data/Yelp/test.tsv",
     "data/Yelp/valid.tsv"]
end

function rawfiles(dataset::Type{SIGDataSet}, config)
    lang = config["lang"]
    th   = lang[1:3]
    ["data/Sigmorphon/task1/all/$(lang)-train-high",
     "data/Sigmorphon/task1/all/$(lang)-test",
     "data/unimorph/$(th)/$(th)"
    ]
end

function rawfiles(dataset::Type{SCANDataSet}, config)
    split, modifier  = config["split"], config["splitmodifier"]
    stsplit = replace(split, "_"=>"")
    if split in ("length","simple")
        ["data/SCAN/$(split)_split/tasks_train_$(stsplit).txt",
         "data/SCAN/$(split)_split/tasks_test_$(stsplit).txt"]
    else
        ["data/SCAN/$(split)_split/tasks_train_$(stsplit)_$(modifier).txt",
         "data/SCAN/$(split)_split/tasks_test_$(stsplit)_$(modifier).txt"]
    end
end

struct Vocabulary{MyDataSet,T}
    tokens::IndexedDict{T}
    inpdict::IndexedDict
    outdict::IndexedDict
    parser::Parser{MyDataSet}
end

Base.length(v::Vocabulary) = length(v.tokens)
collectstr(str::AbstractString) = map(string,collect(str))
function parseDataLine(line::AbstractString, parser::Parser{SIGDataSet})
    lemma, surface, tokens = split(line, '\t')
    (surface=collectstr(surface), lemma=collectstr(lemma), tags=split(tokens,';'))
end

function parseDataLine(line::AbstractString, parser::Parser{SCANDataSet})
    m   = match(parser.regex, line)
    x,y = m.captures
    return (input=split(x,parser.partsSeperator), output=split(y,parser.partsSeperator))
end

function parseDataLine(line::AbstractString, parser::Parser{YelpDataSet})
    #input, proto = split(replace(lowercase(line), r"[^\w\s]"=>s""),"\t")
    input, proto = split(lowercase(strip(line)), '\t')
    return (input=filter(!isempty,split(input,' ')), proto=filter(!isempty,split(proto,' ')))
end

function parseDataFile(f::AbstractString, parser::Parser)
    [parseDataLine(l,parser) for l in eachline(f) if l != ""]
end

function EncodedFormat(a::NamedTuple, v::Vocabulary{SIGDataSet})
    (surface=map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.surface),
     lemma=map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.lemma),
     tags=map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.tags))
end

function EncodedFormat(a::NamedTuple, v::Vocabulary{SCANDataSet})
    (input=map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.input),
     output=map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.output))
end

function EncodedFormat(a::NamedTuple, v::Vocabulary{YelpDataSet})
    input       = map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.input)
    proto       = map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.proto)
    si,sp       = Set(input), Set(proto)
    #limit       = length(v.parser.freewords) + length(specialTokens)
    #I = filter(x->x>limit,setdiff(si,sp)) |> collect
    #D = filter(x->x>limit,setdiff(sp,si)) |> collect
    (input = input,
     proto = proto,
     ID    = inserts_deletes(input,proto))
end

encode(line::NamedTuple, v::Vocabulary) = EncodedFormat(line,v)
encode(data::Vector, v::Vocabulary)     = map(l->encode(l,v),data)

function Vocabulary(data::Vector, parser::Parser{SIGDataSet})
    inpdict, outdict = StrDict(), StrDict()
    for (i,T) in enumerate(specialTokens)
         get!(inpdict,T,i)
         get!(outdict,T,i)
    end

    for (i,(surface, lemma, tags)) in enumerate(data)
        for c in surface
            get!(outdict, c, length(outdict)+1)
        end
        for c in lemma
            get!(inpdict, c, length(inpdict)+1)
        end
        for t in tags
            get!(inpdict,t,length(inpdict)+1)
        end
    end
    tokens = appenddicts(inpdict,outdict)
    Vocabulary{SIGDataSet, String}(IndexedDict(outdict), IndexedDict(inpdict), IndexedDict(outdict), parser)
end

function Vocabulary(data::Vector, parser::Parser{SCANDataSet})
    inpdict, outdict = StrDict(), StrDict()
    for (i,T) in enumerate(specialTokens)
         get!(inpdict,T,i)
         get!(outdict,T,i)
    end
    for (i,(inp,out)) in enumerate(data)
        for w::String in inp
            get!(inpdict, w, length(inpdict)+1)
        end
        for w::String in out
            get!(outdict, w, length(outdict)+1)
        end
    end
    tokens = appenddicts(inpdict,outdict)
    Vocabulary{SCANDataSet, String}(IndexedDict(inpdict), IndexedDict(inpdict), IndexedDict(outdict), parser)
end

function Vocabulary(data::Vector, parser::Parser{YelpDataSet})
    cntdict = StrDict()
    for (i,(inp,out)) in enumerate(data)
        for w::String in inp
            cntdict[w] = get!(cntdict, w, 0) + 1
        end
        for w::String in out
            cntdict[w] = get!(cntdict, w, 0) + 1
        end
    end
    srtdvocab = sort(collect(cntdict), by=x->x[2], rev=true)
    inpdict = StrDict()
    for (i,T) in enumerate(specialTokens)
        get!(inpdict,T,i)
    end
    for (i,T) in enumerate(parser.freewords)
        get!(inpdict,T,length(inpdict)+1)
    end
    println(length(inpdict))
    for (k,v) in srtdvocab
        get!(inpdict, k, length(inpdict) + 1)
        if length(inpdict) == 10000
            break
        end
    end
    Vocabulary{YelpDataSet, String}(IndexedDict(inpdict), IndexedDict(inpdict), IndexedDict(inpdict), parser)
end


xfield(::Type{SIGDataSet},  x, cond::Bool=true) = cond ? [x.lemma;x.tags;[specialIndicies.sep];x.surface] : x.surface
xfield(::Type{SCANDataSet}, x, cond::Bool=true) = cond ? [x.input;[specialIndicies.sep];x.output] : x.input
xfield(::Type{YelpDataSet}, x, cond::Bool=true) = x.input

import StringDistances: compare
compare(x::Vector{Int},    y::Vector{Int},    dist) = compare(String(map(UInt8,x)),String(map(UInt8,y)),dist)
compare(x::Vector{Int},    y::AbstractString, dist) = compare(x,String(map(UInt8,y)),dist)
compare(x::AbstractString, y::Vector{Int},    dist) = compare(String(map(UInt8,x)),y,dist)

function getbatch(iter,B)
    edata = collect(Iterators.take(iter,B))
    if (b = length(edata)) != 0
        sfs,exs, perms = first.(edata), second.(edata), last.(edata)
        r   = sortperm(sfs, by=length, rev=true)
        xi = _pack_sequence(sfs[r])
        xt = map(_pack_sequence,exs[r])
        return (x=xi, examplers=xt, perm=perms[r])
    else
        return nothing
    end
end

function pickprotos(edata)
    x, xp, ID = unzip(edata)
    if first(xp) isa Vector{Int}
        I,D       = unzip(ID)
        #xp = x  # uncomment  if you want to debug with copying. config["kill_edit"] = true
        r         = sortperm(xp, by=length, rev=true)
        x[r], xp[r], I[r], D[r]
    else
        inds = map(l->rand(1:l),length.(xp))
        xp   = map(d->d[1][d[2]],zip(xp,inds))
        ID   = map(d->d[1][d[2]],zip(ID,inds))
        I,D  = unzip(ID)
        r    = sortperm(xp, by=length, rev=true)
        x[r], xp[r], I[r], D[r]
    end
end

function getbatch_proto(iter, B)
    edata   = collect(Iterators.take(iter,B))
    unk, mask, bow, eow = specialIndicies
    if (b = length(edata)) != 0
        d           = pickprotos(edata)
        x, xp, I ,D = d
        x = map(s->(length(s)>25 ? s[1:25] : s) , x) # FIXME: maxlength as constant
        xp_packed   = _pack_sequence(xp)
        x_mask      = get_mask_sequence(length.(x) .+ 2; makefalse=false)
        xp_mask     = get_mask_sequence(length.(xp); makefalse=true)
        xmasked     = PadSequenceArray(map(xi->[bow;xi;eow], x), pad=mask)
        Imask       = get_mask_sequence(length.(I); makefalse=false)
        Imasked     = PadSequenceArray(I, pad=mask)
        Dmask       = get_mask_sequence(length.(D); makefalse=false)
        Dmasked     = PadSequenceArray(D, pad=mask)
        copymask    = create_copy_mask(xp, xp_mask, xmasked)
        return xmasked, x_mask, xp_packed, xp_mask, (I=Imasked, D=Dmasked, Imask=Imask, Dmask=Dmask), copymask, d
    end
    return nothing
end

function create_copy_mask(xp, xp_mask, xmasked)
    mask = trues(size(xmasked,2)-1, size(xp_mask,2), length(xp)) # T x T' x B
    for i=1:size(xmasked,1)
        for t=2:size(xmasked,2)
            token =  xmasked[i,t]
            if token ∉ specialIndicies
                inds  = findall(t->t==token,xp[i])
                mask[t,inds,i] .= false
            end
        end
    end
    return mask
end


#
#
# struct MorphData{T <: AbstractVector}
#     data::T
#     lemma2loc::CrossDict
#     morph2loc::CrossDict
#     num::Int
# end
#
# Base.length(m::MorphData) = length(m.data)
#
# function Base.iterate(m::MorphData, s...)
#     next = iterate(m.data, s...)
#     next === nothing && return nothing
#     (v, snext) = next
#     index = snext-1
#     L = length(m)
#     surface, lemma, tags = v.surface, v.lemma, v.tags
#     ex1 = [sfs for sfs in m.lemma2loc[lemma] if sfs != index]
#     ex2 = [sfs for sfs in m.morph2loc[tags]  if sfs != index]
#     s1 =  map(i->m.data[i].surface, randchoice(ex1, m.num, L))
#     s2 =  map(i->m.data[i].surface, randchoice(ex2, m.num, L))
#     others =  map(i->m.data[i].surface, rand(1:L,2m.num))
#     sc =  [[surface];s1;s2;others]
#     length(sc) == 0 && error()
#     r = sortperm(sc,by=length,rev=true)
#     ex =  sc[r]
#     return (surface, ex, sortperm(r)), snext
# end


#
# function get_neighbours_dict(words, devs...; thresh=0.6, maxcnt=10, lang="unknown", prefix="data/prototype/")
#     dist     = Levenshtein()
#     words    = unique(words)
#     adjlist  = Dict((i,Int[]) for i=1:length(words))
#
#     for i=1:length(words)
#         for j=i+1:length(words)
#             if compare(words[i],words[j],dist) > thresh
#                 push!(adjlist[i],j); push!(adjlist[j],i)
#             end
#         end
#     end
#
#     open(prefix*"train.tsv","w") do f
#         for (i,ns) in adjlist
#             # adjlist[i] = shuffle(ns)[1:min(length(ns),maxcnt)]
#             cnt = 0
#             cw  = words[i]
#             for w in words[shuffle(ns)]
#                 I,D = inserts_deletes(cw,w)
#                 println(f, cw, '\t', w, '\t', join(I), '\t', join(D))
#                 (cnt+=1) == maxcnt && break
#             end
#         end
#     end
#
#     for (k,dwords) in enumerate(devs)
#         f = open(prefix*(k==1 ? "dev.tsv" : "test.tsv"), "w")
#         dwords = unique(dwords)
#         for i=1:length(dwords)
#             cw = dwords[i]
#             cnt = 0
#             for j=randperm(length(words))
#                 w = words[j]
#                 if compare(cw,w,dist) > thresh
#                     I,D = inserts_deletes(cw,w)
#                     length(I) + length(D) == 0 && continue
#                     println(f, cw, '\t', w, '\t', join(I), '\t', join(D))
#                     cnt+=1
#                     cnt == maxcnt && break
#                 end
#             end
#         end
#     end
# end

#
# function encode_proto(data, vocab)
#     map(data) do l
#         map(l) do x
#             vocab.chars[x]
#         end
#     end
# end
#
# struct VocabularyProto
#     chars::IndexedDict{Char}
#     specialTokens::NamedTuple
#     specialIndices::NamedTuple
# end
#
# struct VocabularyProtoSCAN
#     chars::IndexedDict{String}
#     specialTokens::NamedTuple
#     specialIndices::NamedTuple
# end


# function parse_prototype_file_scan(tsvfile, vocab=nothing)
#     data = map(l->split.(split(l, '\t')), eachline(tsvfile))
#     if isnothing(vocab)
#         vocab = Dict{String,Int}()
#         for t in specialTokens; get!(vocab, t,length(vocab)+1); end
#         specialIndices = (unk=1, mask=2, eow=3, bow=4)
#         for l in data
#             for c in first(l)
#                 get!(vocab,c,length(vocab)+1)
#             end
#         end
#         vocab = VocabularyProto(IndexedDict(vocab),specialTokens,specialIndices)
#     end
#     encode_proto(data,vocab), vocab
# end


# function parse_prototype_file(tsvfile, vocab=nothing)
#     data = map(l->collect.(split(l, '\t')), eachline(tsvfile))
#     if isnothing(vocab)
#         vocab = Dict{Char,Int}()
#         for t in specialTokens; get!(vocab, first(t),length(vocab)+1); end
#         specialIndices = (unk=1, mask=2, eow=3, bow=4)
#         for l in data
#             for c in first(l)
#                 get!(vocab,c,length(vocab)+1)
#             end
#         end
#         vocab = VocabularyProto(IndexedDict(vocab),specialTokens,specialIndices)
#     end
#     encode_proto(data,vocab), vocab
# end
#

#
#
# """
#      parseSCANLine(line::AbstractString, p::Parser{<:MyDataSet}; wLemma=true, parseAll)
#      It parses a line from given dataset. Returns a ParsedIO object.
# """
# function parseSCANLine(line::AbstractString, task=:SCAN)
#     if task == :SCAN
#         rx  = r"^IN\:\s(.*?)\sOUT\: (.*?)$"
#         m   = match(rx, line)
#         x,y = m.captures
#         return (in=split(x), out=split(y))
#     end
#     return (in=[],out=[])
# end
#
# struct VocabularySCAN
#     inwords::IndexedDict{String}
#     outwords::IndexedDict{String}
#     chars::IndexedDict{String}
#     specialTokens::NamedTuple
#     specialIndices::NamedTuple
# end
# """
#     EncodedSCANFormat
#     EncodedSCANFormat is structure that keeps morphological anlysis of a word with encoded fields.
# """
# struct EncodedSCANFormat
#     in::Vector{Int}
#     out::Vector{Int}
# end
# function EncodedFormat(a::NamedTuple, v::VocabularySCAN)
#     EncodedSCANFormat([map(x->get(v.chars, x, v.specialIndices.unk)::Int, a.in); v.specialIndices.mask],
#                        map(x->get(v.chars, x, v.specialIndices.unk)::Int, a.out))
# end
# encode(s::NamedTuple, v::VocabularySCAN) = EncodedFormat(s,v)
#
#
# function VocabularySCAN(data::Vector)
#     in2loc, out2loc= StrDict(), StrDict()
#     for (i,T) in enumerate(specialTokens)
#          get!(in2loc,T,i)
#          get!(out2loc,T,i)
#     end
#     specialIndicies = (unk=1, mask=2, eow=3, bow=4)
#
#     for (i,(inp,out)) in enumerate(data)
#         for w::String in inp
#             get!(in2loc, w, length(in2loc)+1)
#         end
#         for w::String in out
#             get!(out2loc, w, length(out2loc)+1)
#         end
#     end
#     words = appenddicts(in2loc,out2loc)
#     VocabularySCAN(IndexedDict(in2loc), IndexedDict(out2loc), IndexedDict(words), specialTokens, specialIndicies)
# end
#
# struct SCANData{T <: AbstractVector}
#     data::T
#     out2loc::CrossDict
#     num::Int
# end
#
# function encode(data::Vector, v::VocabularySCAN, config::Dict)
#     out2loc = CrossDict()
#     edata = Array{EncodedSCANFormat,1}(undef,length(data))
#     for (i,datum) in enumerate(data)
#           encoded = edata[i] = encode(datum,v)
#           push!(get!(out2loc, encoded.out, Int[]),i)
#     end
#     SCANData(edata,out2loc,config["num_examplers"])
# end
#
# inout(v) = [v.in;v.out]
# function Base.iterate(m::SCANData, s...)
#     next = iterate(m.data, s...)
#     next === nothing && return nothing
#     (v, snext) = next
#     index = snext-1
#     L = length(m)
#     input,output = v.in, v.out
#     ex1    = [sfs for sfs in m.out2loc[output] if sfs != index]
#     s1     =  map(i->inout(m.data[i]), randchoice(ex1, m.num, L))
#     s2     =  s1
#     others =  map(i->inout(m.data[i]), rand(1:L,2m.num))
#     sc     =  [[inout(v)];s1;s2;others]
#     length(sc) == 0 && error()
#     r = sortperm(sc,by=length,rev=true)
#     ex =  sc[r]
#     return (inout(v), ex, sortperm(r)), snext
# end
# Base.length(m::SCANData) = length(m.data)
# function trim(chars::Vector{Int},vocab::VocabularySCAN)
#     unk, mask, eow, bow = specialIndicies
#     out = Int[]
#     for c in chars
#         c == vocab.specialIndices.eow && break
#         if c ∉ (unk,bow)
#             push!(out,c)
#         end
#     end
#     return join(vocab.chars[out],' ')
# end


# read tsv
# create char vocab
# split x,xp,I,D
# return array



# Base.length(m::PrototypeData) = length(m.data)

# function Base.iterate(m::MorphData, s...)
#     next = iterate(m.data, s...)
#     next === nothing && return nothing
#     (v, snext) = next
#     index = snext-1
#     L = length(m)
#     surface, lemma, tags = v.surface, v.lemma, v.tags
#     ex1 = [sfs for sfs in m.lemma2loc[lemma] if sfs != index]
#     ex2 = [sfs for sfs in m.morph2loc[tags]  if sfs != index]
#     s1 =  map(i->m.data[i].surface, randchoice(ex1, m.num, L))
#     s2 =  map(i->m.data[i].surface, randchoice(ex2, m.num, L))
#     others =  map(i->m.data[i].surface, rand(1:L,4))
#     sc =  [[surface];s1;s2;others]
#     length(sc) == 0 && error()
#     r = sortperm(sc,by=length,rev=true)
#     ex =  sc[r]
#     return (surface, ex, sortperm(r)), snext
# end



# Create Struct for prototype data array
# Write batch iterator for prototype data array



# struct EncodedFormat
#     surface::Vector{Int}
#     lemma::Vector{Int}
#     tags::Vector{Int}
# end
