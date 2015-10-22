abstract Kernel{T}

abstract StandardKernel{T<:AbstractFloat} <: Kernel{T}

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

call{T<:AbstractFloat}(κ::Kernel{T}, x::T, y::T) = kernel(κ, x, y)
call{T<:AbstractFloat}(κ::Kernel{T}, x::Vector{T}, y::Vector{T}) = kernel(κ, x, y)
call{T<:AbstractFloat}(κ::Kernel{T}, x::Matrix{T}, y::Matrix{T}) = kernel(κ, x, y)


#==========================================================================
  Base Kernels
==========================================================================#

abstract BaseKernel{T<:AbstractFloat} <: StandardKernel{T}

include("kernels/additivekernels.jl")

for kernel in concrete_subtypes(AdditiveKernel)
    kernel_sym = kernel.name.name  # symbol for kernel
    for parent in supertypes(kernel)
        parent_sym = parent.name.name  # symbol for abstract supertype
        @eval begin
            function convert{T<:AbstractFloat}(::Type{$parent_sym{T}}, κ::$kernel_sym)
                convert($kernel_sym{T}, κ)
            end
        end
    end
end

immutable ARD{T<:AbstractFloat} <: BaseKernel{T}
    k::AdditiveKernel{T}
    w::Vector{T}
    function ARD(κ::AdditiveKernel{T}, w::Vector{T})
        all(w .> 0) || throw(ArgumentError("Weight vector w must consist of positive values."))
        new(κ, w)
    end
end
ARD{T<:AbstractFloat}(κ::AdditiveKernel{T}, w::Vector{T}) = ARD{T}(κ, w)

ismercer(κ::ARD) = ismercer(κ.k)
isnegdef(κ::ARD) = isnegdef(κ.k)

rangemax(κ::ARD) = rangemax(κ.k)
rangemin(κ::ARD) = rangemax(κ.k)
attainsrangemax(κ::ARD) = attainsrangemax(κ.k)
attainsrangemin(κ::ARD) = attainsrangemax(κ.k)

function description_string{T<:AbstractFloat,}(κ::ARD{T}, eltype::Bool = true)
    "ARD" * (eltype ? "{$(T)}" : "") * "(κ=$(description_string(κ.k, false)),w=$(κ.w))"
end

convert{T<:AbstractFloat}(::Type{ARD{T}}, κ::ARD) = ARD(convert(Kernel{T},κ.k), convert(Vector{T}, κ.w))


#==========================================================================
  Composite Kernel
==========================================================================#

abstract CompositeKernel{T<:AbstractFloat} <: StandardKernel{T}

include("kernels/compositekernels.jl")

for kernel in concrete_subtypes(CompositeKernel)
    kernel_sym = kernel.name.name  # symbol for concrete kernel type

    field_conversions = [:(convert(Kernel{T}, κ.k))]

    if length(fieldnames(kernel)) != 1
        append!(field_conversions, [:(convert(T, κ.$field)) for field in fieldnames(kernel)[2:end]])
    end

    constructor = Expr(:call, kernel_sym, field_conversions...)

    @eval begin
        convert{T<:AbstractFloat}(::Type{$kernel_sym{T}}, κ::$kernel_sym) = $constructor
    end

    for parent in supertypes(kernel)
        parent_sym = parent.name.name  # symbol for abstract supertype

        @eval begin
            function convert{T<:AbstractFloat}(::Type{$parent_sym{T}}, κ::$kernel_sym)
                convert($kernel_sym{T}, κ)
            end
        end
    end
end


#===================================================================================================
  Composite Kernels
===================================================================================================#

abstract CombinationKernel{T<:AbstractFloat} <: Kernel{T}

immutable KernelProduct{T<:AbstractFloat} <: CombinationKernel{T}
    a::T
    k::Vector{Kernel{T}}
    function KernelProduct(a::T, κ::Vector{Kernel{T}})
        a > 0 || throw(ArgumentError("a = $(a) must be greater than zero."))
        if length(κ) > 1
            all(ismercer, κ) || throw(ArgumentError("All kernels must be Mercer for closure under multiplication."))
        end
        new(a, κ)
    end
end
KernelProduct{T<:AbstractFloat}(a::T, κ::Vector{Kernel{T}}) = KernelProduct{T}(a, κ)

immutable KernelSum{T<:AbstractFloat} <: CombinationKernel{T}
    a::T
    k::Vector{Kernel{T}}
    function KernelSum(a::T, κ::Vector{Kernel{T}})
        a >= 0 || throw(ArgumentError("a = $(a) must be greater than or equal to zero."))
        all(ismercer, κ) || all(isnegdef, κ) || throw(ArgumentError("All kernels must be Mercer or negative definite for closure under addition"))
        new(a, κ)
    end
end
KernelSum{T<:AbstractFloat}(a::T, κ::Vector{Kernel{T}}) = KernelSum{T}(a, κ)

for (kernel_object, kernel_op, kernel_array_op, identity) in (
        (:KernelProduct, :*, :prod, :1),
        (:KernelSum,     :+, :sum,  :0)
    )
    @eval begin

        function $kernel_object(a::Real, κ::Kernel...)
            U = promote_type(typeof(a), map(eltype, κ)...)
            $kernel_object{U}(convert(U, a), Kernel{U}[κ...])
        end

        convert{T<:AbstractFloat}(::Type{$kernel_object{T}}, ψ::$kernel_object) = $kernel_object(convert(T, ψ.a), Kernel{T}[ψ.k...])
        convert{T<:AbstractFloat}(::Type{CombinationKernel{T}}, ψ::$kernel_object) = $kernel_object(convert(T, ψ.a), Kernel{T}[ψ.k...])
        convert{T<:AbstractFloat}(::Type{Kernel{T}}, ψ::$kernel_object) = $kernel_object(convert(T, ψ.a), Kernel{T}[ψ.k...])

        ismercer(ψ::$kernel_object) = all(ismercer, ψ.k)
        isnegdef(ψ::$kernel_object) = all(isnegdef, ψ.k)

        function description_string{T<:AbstractFloat}(ψ::$kernel_object{T}, eltype::Bool = true)
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
