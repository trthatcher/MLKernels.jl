function concrete_subtypes(T::DataType)
    ST = subtypes(T)
    if length(ST) > 0
        vcat(map(concrete_subtypes, ST)...)
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
