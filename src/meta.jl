# For each symbol in symbols, swap the array access
#     ex. :A in symbols => map A[i,j] -> A[j,i]
function transpose_access(symbols, ex)
    if isa(ex, Expr)
        if ex.head == :ref && ex.args[1] in symbols
            @assert length(ex.args) == 3
            :($(ex.args[1])[$(ex.args[3]),$(ex.args[2])])
        else
            Expr(ex.head, map(arg->transpose_access(symbols, arg), ex.args)...)
        end
    else
        ex
    end
end

# Generate a second branch of code with array access transposed for listed symbols
#     cond: an expression representing a condition
#     symbols: the tuple of variables for which array access should be swapped
#     block: the default block of code
macro transpose_access(cond, symbols, block)
    @assert symbols.head == :tuple
    symbollist = symbols.args
    quote
        if $(cond)
            $((transpose_access(symbollist, block)))  # Transpose access if cond is true
        else
            $((block))
        end
    end
end


function concretesubtypes(T::DataType)
    ST = subtypes(T)
    if length(ST) > 0
        vcat(map(concretesubtypes, ST)...)
    elseif !T.abstract # only collect concrete types
        [T]
    else
        []
    end
end

function supertypes(T::DataType)
    ancestors = DataType[]
    S = super(T)
    while S !== Any
        push!(ancestors, S)
        S = super(S)
    end
    ancestors
end

