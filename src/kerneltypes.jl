#===================================================================================================
  Generic Kernels
===================================================================================================#

abstract Kernel{T<:FloatingPoint}

call(κ::Kernel, args...) = kernel_function(κ, args...)

isposdef_kernel(κ::Kernel) = false
is_euclidean_distance(κ::Kernel) = false
is_scalar_product(κ::Kernel) = false

abstract SimpleKernel{T<:FloatingPoint} <: Kernel{T}
abstract CompositeKernel{T<:FloatingPoint} <: Kernel{T}


#===========================================================================
  Standard Kernels
===========================================================================#

include("standardkernels.jl")  # Specific kernels from ML literature


#===========================================================================
  Scaled Kernel
===========================================================================#

type ScaledKernel{T<:FloatingPoint} <: SimpleKernel{T}
    a::T
    κ::StandardKernel{T}
    function ScaledKernel(a::T, κ::StandardKernel{T})
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ)
    end
end
ScaledKernel{T<:FloatingPoint}(a::T, κ::StandardKernel{T}) = ScaledKernel{T}(a, κ)

function convert{T<:FloatingPoint}(::Type{ScaledKernel{T}}, ψ::ScaledKernel) 
    ScaledKernel(convert(T, ψ.a), convert(StandardKernel{T}, ψ.κ))
end

function convert{T<:FloatingPoint}(::Type{SimpleKernel{T}}, ψ::ScaledKernel) 
    ScaledKernel(convert(T, ψ.a), convert(StandardKernel{T}, ψ.κ))
end

@inline function kernel_function{T<:FloatingPoint}(ψ::ScaledKernel{T}, x::Vector{T}, y::Vector{T})
    ψ.a * kernel_function(ψ.κ, x, y)
end

description_string(ψ::ScaledKernel) = "$(ψ.a) * " * description_string(ψ.κ)
isposdef_kernel(ψ::ScaledKernel) = isposdef_kernel(ψ.κ)

function show(io::IO, ψ::ScaledKernel)
    print(io, description_string(ψ))
end

*{T<:FloatingPoint}(a::T, κ::StandardKernel{T}) = ScaledKernel(a, deepcopy(κ))
function *{T<:Real,S<:FloatingPoint}(a::T, κ::StandardKernel{S})
    U = promote_type(T, S)
    *(convert(U, a), convert(Kernel{U}, κ))
end
*(κ::StandardKernel, a::Real) = *(a, κ)

*{T<:FloatingPoint}(a::T, ψ::ScaledKernel{T}) = ScaledKernel{T}(a * ψ.a, deepcopy(ψ.κ))
function *{T<:Real,S<:FloatingPoint}(a::T, κ::ScaledKernel{S})
    U = promote_type(T, S)
    *(convert(U, a), convert(Kernel{U}, κ))
end
*(ψ::ScaledKernel, a::Real) = *(a, ψ)


#===================================================================================================
  Composite Kernels
===================================================================================================#

#== Mercer Kernel Product ====================#

#=

type KernelProduct <: CompositeKernel
    a::Real
    ψ₁::ScalableKernel
    ψ₂::ScalableKernel
    function KernelProduct(a::Real, ψ₁::ScalableKernel, ψ₂::ScalableKernel)
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, ψ₁, ψ₂)
    end
end

@inline function kernel_function(ψ::KernelProduct)
if ψ.a == 1
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = (
        kernel_function(ψ.κ₁)(x, y) * kernel_function(ψ.κ₂)(x, y))
end
k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = (
    kernel_function(ψ.κ₁)(x, y) * kernel_function(ψ.κ₂)(x, y) * convert(T, ψ.a))
end

isposdef_kernel(ψ::KernelProduct) = isposdef_kernel(ψ.κ₁) & isposdef_kernel(ψ.κ₂)

function description_string(ψ::KernelProduct) 
    if ψ.a == 1
        return description_string(ψ.ψ₁) * " * " * description_string(ψ.ψ₂)
    end
    "$(ψ.a) * " * description_string(ψ.ψ₁) * " * " * description_string(ψ.ψ₂)
end

function show(io::IO, ψ::KernelProduct)
    println(io, "Mercer Kernel Product:")
    print(io, " " * description_string(ψ))
end

*(κ₁::ScalableKernel, κ₂::ScalableKernel) = (
    KernelProduct(1, deepcopy(κ₁), deepcopy(κ₂)))

*(κ::ScalableKernel, ψ::ScaledKernel) = (
    KernelProduct(ψ.a, deepcopy(κ), deepcopy(ψ.κ)))

*(ψ::ScaledKernel, κ::ScalableKernel) = (
    KernelProduct(ψ.a, deepcopy(ψ.κ), deepcopy(κ)))

*(ψ₁::ScaledKernel, ψ₂::ScaledKernel) = (
    KernelProduct(ψ₁.a * ψ₂.a, deepcopy(ψ₁.κ), deepcopy(ψ₂.κ)))

*(a::Real, ψ::KernelProduct) = (
    KernelProduct(a * ψ.a, deepcopy(ψ.ψ₁), deepcopy(ψ.ψ₂)))

*(ψ::KernelProduct, a::Real) = a * ψ

=#

#== Mercer Kernel Sum ====================#

#=
type KernelSum <: CompositeKernel
    ψ₁::SimpleKernel
    ψ₂::SimpleKernel
    KernelSum(ψ₁::SimpleKernel, ψ₂::SimpleKernel) = new(ψ₁,ψ₂)
end

@inline function kernel_function(ψ::KernelSum)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = (
        kernel_function(ψ.ψ₁)(x, y) + kernel_function(ψ.ψ₂)(x, y))
end

isposdef_kernel(ψ::KernelSum) = isposdef_kernel(ψ.κ₁) | isposdef_kernel(ψ.κ₂)

function show(io::IO, ψ::KernelSum)
    println(io, "Mercer Kernel Sum:")
    print(io, " " * description_string(ψ.ψ₁) * " + " * description_string(ψ.ψ₂))
end

+(ψ₁::SimpleKernel, ψ₂::SimpleKernel) = KernelSum(deepcopy(ψ₁), deepcopy(ψ₂))

*(a::Real, ψ::KernelSum) = (a * ψ.ψ₁) + (a * ψ.ψ₂)
*(ψ::KernelSum, a::Real) = a * ψ

=#

