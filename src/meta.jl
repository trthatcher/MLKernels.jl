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

macro transpose_access(cond, symbols, block)
    @assert symbols.head == :tuple
    symbollist = symbols.args
    return quote
        if $(cond)
            $((transpose_access(symbollist, block)))
        else
            $((block))
        end
    end
end

