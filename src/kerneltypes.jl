#===================================================================================================
  Generic Kernels
===================================================================================================#

abstract Kernel{T<:FloatingPoint}

#eltype{T}(κ::Kernel{T}) = T

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

function ScaledKernel{T<:Real,S}(a::T, κ::StandardKernel{S})
    U = promote_type(T, S)
    ScaledKernel(convert(U, a), convert(Kernel{U}, κ))
end

for kernel_type in (:ScaledKernel, :SimpleKernel, :Kernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, ψ::ScaledKernel) 
            ScaledKernel(convert(T, ψ.a), convert(Kernel{T}, ψ.κ))
        end
    end
end

@inline function kernel_function{T<:FloatingPoint}(ψ::ScaledKernel{T}, x::Vector{T}, y::Vector{T})
    ψ.a * kernel_function(ψ.κ, x, y)
end

description_string(ψ::ScaledKernel) = "ScaledKernel($(ψ.a), " * description_string(ψ.κ) * ")"
isposdef_kernel(ψ::ScaledKernel) = isposdef_kernel(ψ.κ)

function show(io::IO, ψ::ScaledKernel)
    print(io, description_string(ψ))
end

*(a::Real, κ::StandardKernel) = ScaledKernel(a, deepcopy(κ))
*(κ::StandardKernel, a::Real) = *(a, κ)

*(a::Real, ψ::ScaledKernel) = ScaledKernel(a * ψ.a, deepcopy(ψ.κ))
*(ψ::ScaledKernel, a::Real) = *(a, ψ)


#===========================================================================
  Product Kernel
===========================================================================#

type KernelProduct{T<:FloatingPoint} <: CompositeKernel{T}
    a::T
    κ₁::StandardKernel{T}
    κ₂::StandardKernel{T}
    function KernelProduct(a::T, κ₁::StandardKernel{T}, κ₂::StandardKernel{T})
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ₁, κ₂)
    end
end
function KernelProduct{T<:FloatingPoint}(a::T, κ₁::StandardKernel{T}, κ₂::StandardKernel{T})
    KernelProduct{T}(a, κ₁, κ₂)
end

function KernelProduct{T<:Real,S,U}(a::T, κ₁::StandardKernel{S}, κ₂::StandardKernel{U})
    V = promote_type(T, S, U)
    KernelProduct(convert(V, a), convert(Kernel{V}, κ₁), convert(Kernel{V}, κ₂))
end

for kernel_type in (:KernelProduct, :CompositeKernel, :Kernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, ψ::KernelProduct) 
            KernelProduct(convert(T, ψ.a), convert(Kernel{T}, ψ.κ₁),  convert(Kernel{T}, ψ.κ₂))
        end
    end
end

@inline function kernel_function{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T})
    a * kernel_function(ψ.κ₁, x, y) * kernel_function(ψ.κ₂, x, y)
end

isposdef_kernel(ψ::KernelProduct) = isposdef_kernel(ψ.κ₁) & isposdef_kernel(ψ.κ₂)

function description_string{T<:FloatingPoint}(ψ::KernelProduct{T}) 
    "ProductKernel{$(T)}($(ψ.a), " * description_string(ψ.κ₁) * ", " * (
    description_string(ψ.κ₂) * ")")
end

function show(io::IO, ψ::KernelProduct)
    print(io, description_string(ψ))
end

function *{T<:FloatingPoint,S<:FloatingPoint}(κ₁::StandardKernel{T}, κ₂::StandardKernel{S})
    KernelProduct(one(promote_type(T, S)), deepcopy(κ₁), deepcopy(κ₂))
end

*(κ::StandardKernel, ψ::ScaledKernel) = KernelProduct(ψ.a, deepcopy(κ), deepcopy(ψ.κ))
*(ψ::ScaledKernel, κ::StandardKernel) = *(κ, ψ)

*(ψ₁::ScaledKernel, ψ₂::ScaledKernel) = KernelProduct(ψ₁.a*ψ₂.a, deepcopy(ψ₁.κ), deepcopy(ψ₂.κ))

*(a::Real, ψ::KernelProduct) = KernelProduct(a * ψ.a, deepcopy(ψ.ψ₁), deepcopy(ψ.ψ₂))
*(ψ::KernelProduct, a::Real) = *(a, ψ)


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

