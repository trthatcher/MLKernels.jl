function isconcretetype(T::DataType)
    try
        T() # fails if T is concrete but does not provide a no-argument default constructor
        return true
    catch
        return false
    end
end

function subtypeleaves(T::DataType)
    ST = subtypes(T)
    if length(ST) > 0
        vcat(map(subtypeleaves, ST)...)
    else
        T
    end
end

function supertypes(T::DataType)
    result = DataType[]
    S = super(T)
    while S !== Any
        push!(result, S)
        S = super(S)
    end
    result
end

for kernelobject in filter(isconcretetype, subtypeleaves(StandardKernel))
    kernelobjectname = kernelobject.name.name # symbol for concrete kernel type

    fieldconversions = [:(convert(T, κ.$field)) for field in names(kernelobject)]
    constructorcall = Expr(:call, kernelobjectname, fieldconversions...)

    @eval begin
        convert{T<:FloatingPoint}(::Type{$kernelobjectname{T}}, κ::$kernelobjectname) = $constructorcall
    end

    for kerneltype in supertypes(kernelobject)
        kerneltypename = kerneltype.name.name # symbol for abstract supertype

        @eval begin
            function convert{T<:FloatingPoint}(::Type{$kerneltypename{T}}, κ::$kernelobjectname)
                convert($kernelobjectname{T}, κ)
            end
        end
    end
end

