using LibGit2, Random, KnetLayers, StringDistances
import KnetLayers: IndexedDict, _pack_sequence

const specialTokens   = (unk="<unk>", mask="<pad>", eow="</s>", bow="<s>", sep="→", iosep="#", copy="<copy>")
const specialIndicies = (unk=1, mask=2, eow=3, bow=4, sep=5, iosep=6, copy=7)

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
    joinpath(defaultpath(SIGDataSet), string(opt["model"],"_",opt["task"],"_",opt["lang"],"_condition_",opt["conditional"],"_",opt["split"]))

prefix(dataset::Type{YelpDataSet}, opt) =
    joinpath(defaultpath(YelpDataSet), string(opt["model"], "_", opt["task"]))

function rawfiles(dataset::Type{YelpDataSet}, config)
    ["data/Yelp/train.tsv",
     "data/Yelp/test2.tsv",
     "data/Yelp/valid.tsv"]
end

function rawfiles(dataset::Type{SIGDataSet}, config)
    lang = config["lang"]
    th   = lang[1:3]
    split = config["split"]
    ["data/Sigmorphon/task1/all/$(lang)-train-$(split)",
     "data/Sigmorphon/task1/all/$(lang)-test"
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
    proto, input = split(lowercase(strip(line)), '\t')
    return (X=filter(!isempty,split(input,' ')), xp=filter(!isempty,split(proto,' ')))
end

function parseDataFile(f::AbstractString, parser::Parser)
    [parseDataLine(l,parser) for l in progress(eachline(f)) if l != ""]
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

function encode_cond(a::NamedTuple, v::Vocabulary)
    (input=map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.input),
     output=map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.output))
end

function EncodedFormat(a::NamedTuple, v::Vocabulary{YelpDataSet})
    input       = map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.input)
    proto       = map(x->get(v.tokens, x, specialIndicies.unk)::Int, a.proto)
    si,sp       = Set(input), Set(proto)
    (x  = input,
     xp = proto,
     ID = inserts_deletes(input,proto))
end

encode(line::NamedTuple, v::Vocabulary) = EncodedFormat(line,v)
encode(data::Vector, v::Vocabulary)     = map(l->encode(l,v), data)

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
    Vocabulary{SIGDataSet, String}(IndexedDict(tokens), IndexedDict(inpdict), IndexedDict(outdict), parser)
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
    Vocabulary{SCANDataSet, String}(IndexedDict(tokens), IndexedDict(inpdict), IndexedDict(outdict), parser)
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

function xfield(::Type{SIGDataSet},  x, cond::Bool=true; masktags::Bool=false, subtask="reinflection")
    tags = masktags ? fill!(similar(x.tags),specialIndicies.mask) : x.tags
    if subtask=="reinflection"
        cond ? [x.lemma;tags;[specialIndicies.sep];x.surface] : [x.lemma;tags]
    else
        cond ? [x.surface;[specialIndicies.sep];x.lemma;x.tags] : x.surface
    end
end
xfield(::Type{SCANDataSet}, x, cond::Bool=true) = cond ? [x.input;[specialIndicies.sep];x.output] : x.input
xfield(::Type{YelpDataSet}, x, cond::Bool=true) = x.input

xy_field(::Type{SCANDataSet}, x, subtask=nothing)   = x

xy_field(::Type{SIGDataSet},  x, subtask="analyses") =
    subtask == "analyses" ? (input=x.surface, output=[x.lemma;x.tags]) : (input=[x.lemma;x.tags], output=x.surface)

# To save regex field in the parser.
function _ser(x::Vocabulary{T},s::IdDict,::typeof(JLDMODE)) where T
    if !haskey(s,x)
        if isa(x.parser.regex,Nothing) && isa(x.parser.partsSeperator,Nothing)
            s[x] = Vocabulary(x.tokens, x.inpdict, x.outdict, Parser{T}())
        else
            s[x] = Vocabulary(x.tokens, x.inpdict, x.outdict,  Parser{T}(ntuple(i->nothing,4)...))
        end
    end
    return s[x]
end
