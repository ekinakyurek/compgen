using LibGit2, Random, Knet, KnetLayers, StringDistances
import KnetLayers: IndexedDict, _pack_sequence
import Knet: _ser, JLDMODE, save
dir(path...) = joinpath(pwd(),path...)
const specialTokens   = (unk="<unk>", mask="<pad>", eow="</s>", bow="<s>", sep="→", iosep="#", copy="<copy>")
const specialIndicies = (unk=1, mask=2, eow=3, bow=4, sep=5, iosep=6, copy=7)

abstract type DataSet; end
abstract type SIGDataSet <: DataSet; end
abstract type SCANDataSet <: DataSet; end
abstract type YelpDataSet <: DataSet; end

const CHECKPOINT_DIR = get(ENV,"RECOMB_CHECKPOINT_DIR","./checkpoints/")
const DATA_DIR = get(ENV,"RECOMB_DATA_DIR","./data/")

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

defaultpath(dataset::Type{SIGDataSet})  = dir(DATA_DIR,get(ENV,"RECOMB_SIG_SUBDIR","SIGDataSet"))
defaultpath(dataset::Type{SCANDataSet}) = dir(DATA_DIR,get(ENV,"RECOMB_SCAN_SUBDIR","SCANDataSet"))
defaultpath(dataset::Type{YelpDataSet}) = dir(DATA_DIR,get(ENV,"RECOMB_YELP_SUBDIR","Yelp"))

function download(dataset::Type{SIGDataSet}; path=defaultpath(SIGDataSet))
    @show path
    !isdir(path) && LibGit2.clone("https://github.com/sigmorphon/conll2018", path)
    # unipath = "data/unimorph"
    # if !isdir(unipath)
    #     mkdir(unipath)
    #     server = "https://github.com/unimorph/"
    #     for lang in ("eng", "spa", "tur")
    #         langpath = joinpath(unipath,lang)
    #         !isdir(langpath) && LibGit2.clone(server*lang, joinpath(langpath))
    #     end
    # end
    return true
end

function download(dataset::Type{SCANDataSet}; path=defaultpath(SCANDataSet))
    @show path
    !isdir(path) ? LibGit2.clone("https://github.com/brendenlake/SCAN", path) : true
end

function download(dataset::Type{YelpDataSet}; path=defaultpath(YelpDataSet))
    if !isdir("$DATA_DIR/Glove")
        download("http://nlp.stanford.edu/data/glove.6B.zip","$DATA_DIR/glove.6B.zip")
        run(`unzip -qq $DATA_DIR/glove.6B.zip -d $DATA_DIR/Glove/`)
        rm("$DATA_DIR/glove.6B.zip")
    else
        @warn "There is already $DATA_DIR/Glove folder, skipping..."
    end
    if !isdir("$(defaultpath(YelpDataSet))")
        mkdir("$(defaultpath(YelpDataSet))")
        server = "https://worksheets.codalab.org/rest/bundles/0x984fe19b60f1479b925933eacbfda8d8/contents/blob/"
        for file in ["train","test","valid"] .*  ".tsv"
            Base.download(joinpath(server,file), joinpath(path,file))
        end
        Base.download(joinpath(server,"free.txt"),  joinpath(path,"free.txt"))
    else
        @warn "There is already $(defaultpath(YelpDataSet)) folder, skipping..."
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
    ["$(defaultpath(YelpDataSet))/train.tsv",
     "$(defaultpath(YelpDataSet))/test2.tsv",
     "$(defaultpath(YelpDataSet))/valid.tsv"]
end

function rawfiles(dataset::Type{SIGDataSet}, config)
    lang = config["lang"]
    th   = lang[1:3]
    split = config["split"]
    ["$(defaultpath(SIGDataSet))/task1/all/$(lang)-train-$(split)",
     "$(defaultpath(SIGDataSet))/task1/all/$(lang)-test"
    ]
end

function rawfiles(dataset::Type{SCANDataSet}, config)
    split, modifier  = config["split"], config["splitmodifier"]
    stsplit = replace(split, "_"=>"")
    if split in ("length","simple")
        ["$(defaultpath(SCANDataSet))/$(split)_split/tasks_train_$(stsplit).txt",
         "$(defaultpath(SCANDataSet))/$(split)_split/tasks_test_$(stsplit).txt"]
    else
        ["$(defaultpath(SCANDataSet))/$(split)_split/tasks_train_$(stsplit)_$(modifier).txt",
         "$(defaultpath(SCANDataSet))/$(split)_split/tasks_test_$(stsplit)_$(modifier).txt"]
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



function preprocess_json_format(neighboorhoods, splits, esets, edata; subtask="reinflection")
    processed = []
    for (set,inds) in zip(esets,splits)
        proc_set = []
        for (d,i) in zip(set,inds)
            !haskey(neighboorhoods,string(i-1)) && continue
            x  = xfield(SIGDataSet,d,true; subtask=subtask)
            for ns in neighboorhoods[string(i-1)]
                xp  = xfield(SIGDataSet,edata[ns[1]+1],true; subtask=subtask)
                xpp = xfield(SIGDataSet,edata[ns[2]+1],true; subtask=subtask)
                push!(proc_set, (x=x, xp=xp, xpp=xpp, ID=inserts_deletes(x,xp)))
                #push!(proc_set, (x=x, xp=xpp, xpp=xp, ID=inserts_deletes(x,xpp)))
            end
        end
        push!(processed, proc_set)
    end
    return processed
end


"""
    TRtoLower(x::Char)
    TRtoLower(s::AbstractString)
    lowercase function for Turkish locale
"""
TRtoLower(x::Char) = x=='I' ? 'ı' : lowercase(x);
TRtoLower(s::AbstractString) = map(TRtoLower,s)
function special_lowercase(chars, lcase)
    newchars = similar(chars)
    start = true
    for (i,c) in enumerate(chars)
        if start
            newchars[i] = lcase(c)
        else
            newchars[i] = c
        end
        if c == " "
            newchars[1:i] = map(lcase,newchars[1:i])
            start = true
        else
            start = false
        end
    end
    return newchars
end



function read_from_jsons(path, config; level="hard")
    lang = config["lang"]
    path = path * lang* "/"
    lcase = lang == "turkish" ? TRtoLower : lowercase
    subtask = config["subtask"]
    println("reading from $path ,and subtask $(subtask)")
    hints, seed = config["hints"], config["seed"]
    fix = "hints-$hints.$seed"
    data = map(d->convert(Vector{Int},d) .+ 1, JSON.parsefile(path*"seqs.$fix.json"))
    splits = JSON.parsefile(path*"splits.$fix.json")
    neighbourhoods = JSON.parsefile(path*"neighborhoods.$fix.json")
    vocab = JSON.parsefile(path*"vocab.json")
    vocab = convert(Dict{String,Int},vocab)
    for (k,v) in vocab; vocab[k] = v+1; end
    vocab = IndexedDict(vocab)
    strdata = [split_array(vocab[d][3:end-1],"<sep>") for d in data]
    strdata = map(strdata) do  d
            lemma, tags = split_array(special_lowercase(d[1],lcase),isuppercaseornumeric; include=true)
            (surface=special_lowercase(d[2],lcase), lemma=lemma, tags=tags)
    end
    if isfile(path*"generated.$fix.json")
        #augmented_data = map(d->convert(Vector{Int},d) .+ 1, JSON.parsefile(path*"generated.$fix.json"))
        augmented_data = JSON.parsefile(path*"generated.$fix.json")
        aug = map(augmented_data) do  d
            output, input = map(String,d["inp"][2:end-1]), map(String,d["out"])
            (input=input,output=output)
        #     if length(findall(t->t=="<sep>", x)) == 1
        #         input, output = split_array(x,"<sep>")
        #         input, output  = special_lowercase(input,lcase), special_lowercase(output,lcase)
        #         if any(map(isuppercaseornumeric, x))
        #             lemma, tags   = split_array(input, isuppercaseornumeric; include=true)
        #             (surface=output, lemma=lemma, tags=tags)
        #             (input=output, output=input)
        #         else
        #             nothing
        #         end
        #     else
        #         nothing
        #     end
        end
        aug  = filter!(!isnothing, aug)
    else
        aug  = []
    end
    println(length(aug))
    vocab  = Vocabulary(strdata, Parser{SIGDataSet}())
    edata  = encode(strdata,vocab)
    splits = map(s->Int.(splits[s]) .+ 1,("train","test_hard","test_easy","val_easy","val_hard"))
    esets  = [edata[s] for s in splits]
    #eaug   = encode(aug,vocab)
    processed = preprocess_json_format(neighbourhoods, splits, esets, edata; subtask=config["subtask"])
    if get(config,"modeldir",nothing) == nothing
        model     = config["model"](vocab, config; embeddings=nothing)
    else
        model     = KnetLayers.load(config["modeldir"],"model")
        model.config["modeldir"] = config["modeldir"]
    end
    #model = (vocab=vocab,)
    #processed  = preprocess(model, esets...)
    #save(fname, "data", processed, "esets", esets, "vocab", vocab, "embeddings", nothing)
    return processed, esets, model, aug
end

function print_train_data(config)
    datapath = defaultpath(config["task"])*"/"
    println("reading from $(datapath)")
    processed, esets, model, aug = read_from_jsons(datapath, config)
    vocab = model.vocab
    lang, seed, hints = config["lang"], config["seed"], config["hints"]
    println("lang $lang hints $hints seed $seed")
    for (i,split) in enumerate(("train","test_hard","test_easy","val_easy","val_hard"))
        data = esets[i]
        outpath = defaultpath(config["task"])* "/" * lang * "/$(split).hints-$(hints).$(seed).txt"
        println("writing to $outpath")
        f = open(outpath,"w")
        for d in data
            lemma   = join(vocab.tokens[d.lemma],"")
            surface = join(vocab.tokens[d.surface],"")
            tags    = join(vocab.tokens[d.tags],";")
            println(f,lemma,"\t",surface,"\t",tags)
        end
        close(f)
    end
end

function create_ppl_examples(eset, numex=200)
    test_sentences = unique(map(x->x.x,eset))
    test_sentences = test_sentences[1:min(length(test_sentences),numex)]
    example_inds   = [findall(e->e.x==t,eset) for t in test_sentences]
    data = []; inds = []; crnt = 1
    for ind in example_inds
        append!(data, eset[ind])
        push!(inds, crnt:crnt+length(ind)-1)
        crnt = crnt+length(ind)
    end
    return data,inds
end
