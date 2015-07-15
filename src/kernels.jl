abstract Kernel{T}

abstract StandardKernel{T<:FloatingPoint} <: Kernel{T}

function show(io::IO, κ::Kernel)
    print(io, description_string(κ))
end

eltype{T}(κ::Kernel{T}) = T

ismercer(::Kernel) = false
isnegdef(::Kernel) = false

rangemax(::Kernel) = Inf
rangemin(::Kernel) = -Inf
attainsrangemax(::Kernel) = true
attainsrangemin(::Kernel) = true

<=(κ::Kernel, x::Real) = attainsrangemax(κ) ? (rangemax(κ) <= x) : (rangemax(κ) <= x)
<=(x::Real, κ::Kernel) = attainsrangemin(κ) ? (x <= rangemin(κ)) : (x <  rangemin(κ))

<(κ::Kernel, x::Real)  = attainsrangemax(κ) ? (rangemax(κ) <= x) : (rangemax(κ) <  x)
<(x::Real, κ::Kernel)  = attainsrangemin(κ) ? (x <  rangemin(κ)) : (x <= rangemax(κ))

>=(κ::Kernel, x::Real) = x <= κ
>=(x::Real, κ::Kernel) = κ <= x

>(κ::Kernel, x::Real)  = x < κ
>(x::Real, κ::Kernel)  = κ < x

#==========================================================================
  Base Kernels
==========================================================================#

abstract BaseKernel{T<:FloatingPoint} <: StandardKernel{T}

include("kernels/additivekernels.jl")

immutable ARD{T<:FloatingPoint} <: BaseKernel{T}
    k::AdditiveKernel{T}
    w::Vector{T}
    function ARD(κ::AdditiveKernel{T}, w::Vector{T})
        all(w .> 0) || error("Weights must be positive real numbers.")
        new(κ, w)
    end
end
ARD{T<:FloatingPoint}(κ::AdditiveKernel{T}, w::Vector{T}) = ARD{T}(κ, w)

ismercer(κ::ARD) = ismercer(κ.k)
isnegdef(κ::ARD) = isnegdef(κ.k)

rangemax(κ::ARD) = rangemax(κ.k)
rangemin(κ::ARD) = rangemax(κ.k)
attainsrangemax(κ::ARD) = attainsrangemax(κ.k)
attainsrangemin(κ::ARD) = attainsrangemax(κ.k)

function description_string{T<:FloatingPoint,}(κ::ARD{T}, eltype::Bool = true)
    "ARD" * (eltype ? "{$(T)}" : "") * "(κ=$(description_string(κ.k, false)),w=$(κ.w))"
end

convert{T<:FloatingPoint}(::Type{ARD{T}}, κ::ARD) = ARD(convert(Kernel{T},κ.k), convert(Vector{T}, κ.w))

for kernel in concrete_subtypes(AdditiveKernel)
    kernel_sym = kernel.name.name  # symbol for kernel
    for parent in supertypes(kernel)
        parent_sym = parent.name.name  # symbol for abstract supertype
        @eval begin
            function convert{T<:FloatingPoint}(::Type{$parent_sym{T}}, κ::$kernel_sym)
                convert($kernel_sym{T}, κ)
            end
        end
    end
end


#==========================================================================
  Composite Kernel
==========================================================================#

abstract CompositeKernel{T<:FloatingPoint} <: StandardKernel{T}

include("kernels/compositekernels.jl")

for kernel in concrete_subtypes(CompositeKernel)
    kernel_sym = kernel.name.name  # symbol for concrete kernel type

    field_conversions = [:(convert(Kernel{T}, κ.k))]

    if length(names(kernel)) != 1
        append!(field_conversions, [:(convert(T, κ.$field)) for field in names(kernel)[2:end]])
    end

    constructor = Expr(:call, kernel_sym, field_conversions...)

    @eval begin
        convert{T<:FloatingPoint}(::Type{$kernel_sym{T}}, κ::$kernel_sym) = $constructor
    end

    for parent in supertypes(kernel)
        parent_sym = parent.name.name  # symbol for abstract supertype

        @eval begin
            function convert{T<:FloatingPoint}(::Type{$parent_sym{T}}, κ::$kernel_sym)
                convert($kernel_sym{T}, κ)
            end
        end
    end
end


GaussianKernel{T<:FloatingPoint}(α::T = 1.0) = ExponentialKernel(SquaredDistanceKernel(one(T)), α)
RadialBasisKernel{T<:FloatingPoint}(α::T = 1.0) = ExponentialKernel(SquaredDistanceKernel(one(T)),α)
LaplacianKernel{T<:FloatingPoint}(α::T = 1.0) = ExponentialKernel(SquaredDistanceKernel(one(T)),α, convert(T, 0.5))
LinearKernel{T<:FloatingPoint}(α::T = 1.0, c::T = one(T)) = PolynomialKernel(ScalarProductKernel(), α, c, one(T))


#===========================================================================
  Kernel Conversions
===========================================================================#



#=
function convert{T<:FloatingPoint}(::Type{ARD{T}}, κ::ARD)
    ARD(convert(Kernel{T}, κ.k), T[κ.w...])
end

for kernelobject in concretesubtypes(Kernel)
    kernelobjectsym = kernelobject.name.name  # symbol for concrete kernel type

    for kerneltype in supertypes(kernelobject)
        kerneltypesym = kerneltype.name.name  # symbol for abstract supertype

        @eval begin
            function convert{T<:FloatingPoint}(::Type{$kerneltypesym{T}}, κ::$kernelobjectsym)
                convert($kernelobjectsym{T}, κ)
            end
        end
    end
end
=#

#===================================================================================================
  Composite Kernels
===================================================================================================#

abstract CombinationKernel{T<:FloatingPoint} <: Kernel{T}

#=

for (kernel_object, kernel_op, kernel_array_op, identity) in (
        (:KernelProduct, :*, :prod, :1),
        (:KernelSum,     :+, :sum,  :0)
    )
    @eval begin

        immutable $kernel_object{T<:FloatingPoint} <: CombinationKernel{T}
            a::T
            k::Vector{Kernel{T}}
            function $kernel_object(a::T, κ::Vector{Kernel{T}})
                $(kernel_op == :+ ? :(>=) : :>)(a, 0) || error("a = $(a) must be greater than zero.")
                new(a, κ)
            end
        end
        $kernel_object{T<:FloatingPoint}(a::T, κ::Vector{Kernel{T}}) = $kernel_object{T}(a, κ)

        function $kernel_object(a::Real, κ::Kernel...)
            U = promote_type(typeof(a), map(eltype, κ)...)
            $kernel_object{U}(convert(U, a), Kernel{U}[κ...])
        end

        convert{T<:FloatingPoint}(::Type{$kernel_object{T}}, ψ::$kernel_object) = $kernel_object(convert(T, ψ.a), Kernel{T}[ψ.k...])
        convert{T<:FloatingPoint}(::Type{CombinationKernel{T}}, ψ::$kernel_object) = $kernel_object(convert(T, ψ.a), Kernel{T}[ψ.k...])
        convert{T<:FloatingPoint}(::Type{Kernel{T}}, ψ::$kernel_object) = $kernel_object(convert(T, ψ.a), Kernel{T}[ψ.k...])

        kernel{T<:FloatingPoint}(ψ::$kernel_object{T}, x::KernelInput{T}, y::KernelInput{T}) = $kernel_op(ψ.a, $kernel_array_op(map(κ -> kernel(κ,x,y), ψ.k)))

        ismercer(ψ::$kernel_object) = all(ismercer, ψ.k)
        isnegdef(ψ::$kernel_object) = all(isnegdef, ψ.k)

        function description_string{T<:FloatingPoint}(ψ::$kernel_object{T}, eltype::Bool = true)
            descs = map(κ -> description_string(κ, false), ψ.k)
            if eltype
                $(string(kernel_object)) * (eltype ? "{$(T)}" : "") * "($(ψ.a), $(join(descs, ", ")))"
            else
                if $kernel_op !== (*) && ψ.a != $identity
                    insert!(descs, 1, string(ψ.a))
                end
                desc_str = string("(", join(descs, $(string(" ", kernel_op, " "))), ")")
                if $kernel_op === (*) && ψ.a != $identity
                    string(ψ.a, desc_str)
                else
                    desc_str
                end
            end
        end

        $kernel_op(a::Real, κ::Kernel) = $kernel_object(a, κ)
        $kernel_op(κ::Kernel, a::Real) = $kernel_op(a, κ)

        $kernel_op(a::Real, ψ::$kernel_object) = $kernel_object($kernel_op(a, ψ.a), ψ.k...)
        $kernel_op(ψ::$kernel_object, a::Real) = $kernel_op(a, ψ)

        $kernel_op(κ1::$kernel_object, κ2::$kernel_object) = $kernel_object($kernel_op(κ1.a, κ2.a), κ1.k..., κ2.k...)

        $kernel_op(κ::Kernel, ψ::$kernel_object) = $kernel_object(ψ.a, κ, ψ.k...)
        $kernel_op(ψ::$kernel_object, κ::Kernel) = $kernel_object(ψ.a, ψ.k..., κ)

        $kernel_op(κ1::Kernel, κ2::Kernel) = $kernel_object($identity, κ1, κ2)

    end
end

=#
