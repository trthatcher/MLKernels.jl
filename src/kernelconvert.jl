for kernelobject in concretesubtypes(StandardKernel)
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

