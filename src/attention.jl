struct PositionalAttention
    key_transform
    query_transform
    memory_transform
    aij_k::Union{Embed,Nothing}
    aij_v::Union{Embed,Nothing}
    attdim::Int
    normalize::Bool
end

# Attention Weight Initilization from the paper: forget the reference
function att_winit(output::Int, input::Int; atype=arrtype)
    st = (1 / sqrt(input))
    2st .* rand!(atype(undef,output,input)) .- st
end

function PositionalAttention(;memory::Int, query::Int, att::Int,
                            aij_k=nothing,
                            aij_v=nothing,
                            act=Tanh(), #activation
                            valT=true,
                            queryT=true,
                            keyT=true,
                            normalize=false)
    transforms = map(zip((keyT, queryT, valT),(memory, query, memory))) do (trans,input)
                    if trans
                        Dense(;input=input, output=att, activation=act, winit=att_winit) # winit=linear_init(input), binit=linear_init(input),
                    else
                        NonAct()
                    end
                 end
    PositionalAttention(transforms...,aij_k,aij_v,att, normalize)
end

function attention2d_feature(m::PositionalAttention, memory, query; pdrop=0.0, mask=nothing, positions=nothing)
    tkey     = m.key_transform(memory)
    tquery   = m.query_transform(query)
    scores   = tquery .* tkey
    scores   = dropout(scores, pdrop)
    if m.normalize
        scores = scores ./ sqrt(m.attdim) #[H,B]
    end
    weights  = softmax(scores;dims=1) #[H,B]
    values   = m.memory_transform(memory) .* weights # [H,B]
    return (values, scores, weights)
end

function attention2d(m::PositionalAttention, memory, query; pdrop=0.0, mask=nothing, positions=nothing)
    tquery  = m.query_transform(query)
    #memory  = memory # H,B,T' or H,T',B
    tkey = m.key_transform(memory) # H,B,T'
    scores = mat(sum(expand(tquery, dim=3) .* tkey, dims=1))' #[T' X B]
    if !isnothing(positions)
        scores += At_mul_B(m.aij_k(positions), tquery) # T'X B
    end
    scores = dropout(scores, pdrop) #./ sqrt(m.attdim)
    if m.normalize
        scores = scores ./ sqrt(m.attdim)
    end
    if !isnothing(mask)
        scores = applymask2(scores, mask, +) # size(mask) == [T', B]
    end
    weights = softmax(scores;dims=1) #[T',B]
    values  = mat(sum(m.memory_transform(memory) .* expand(weights',dim=1), dims=3),dims=1) # [H,B]
    if !isnothing(positions)
        values += m.aij_v(positions) * weights
    end
    values, scores, weights
end

function attention3d(m::PositionalAttention, memory, query; pdrop=0.0, mask=nothing, positions=nothing)
    tquery  = m.query_transform(query) # [A,B,T], [A,B] | [A,T,B], [A,B]
    memory = batch_last(memory)
    tkey   = m.key_transform(memory)
    #tquery = batch_last(tquery)
    scores = batch_last(bmm(tkey, batch_last(tquery),transA=true)) #[T',B,T]
    if !isnothing(positions)
        scores += bmm(m.aij_k(positions), tquery; transA=true)
    end
    scores = dropout(scores, pdrop) #./ sqrt(m.attdim)
    if m.normalize
        scores = scores ./ sqrt(m.attdim)
    end
    if !isnothing(mask)
        scores = applymask2(scores, expand(mask, dim=3), +) # size(mask) == [T', B , 1]
    end
    #memory H,B,T' # weights T' X B X T
    weights  = softmax(scores;dims=1) #[T',B,T]
    values   = batch_last(bmm(m.memory_transform(memory),batch_last(weights))) # [H,T,B]
    if !isnothing(positions)
        values += bmm(m.aij_v(positions), weights) # H X B X T times
    end
    values, scores, weights
end

function (m::PositionalAttention)(memory, query; feature=false, o...)
    if feature
        attention2d_feature(m, memory, query; o...) # currently only 2d x 2d
    elseif ndims(query) == 2
        attention2d(m, memory, query; o...)
    else
        attention3d(m, memory, query; o...)
    end
end

clip_index(x; K=16) = max(-K, min(K, x)) + K + 1
position_calc(L, step; K=16) = clip_index.(collect(1:L) .- step; K=K)
