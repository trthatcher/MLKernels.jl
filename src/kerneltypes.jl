#===================================================================================================
  Generic Kernels
===================================================================================================#

abstract Kernel{T<:FloatingPoint}

eltype{T}(κ::Kernel{T}) = T

#call{T<:FloatingPoint}(κ::Kernel{T}, x::Vector{T}, y::Vector{T}) = kernel_function(κ, x, y)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}) = kernel_matrix(κ, X)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}) = kernel_matrix(κ, X, Y)

isposdef_kernel(κ::Kernel) = false

abstract SimpleKernel{T<:FloatingPoint} <: Kernel{T}
abstract CompositeKernel{T<:FloatingPoint} <: Kernel{T}


#===========================================================================
  Standard Kernels
===========================================================================#

include("standardkernels.jl")  # Specific kernels from ML literature


#===========================================================================
  Scaled Kernel
===========================================================================#

immutable ScaledKernel{T<:FloatingPoint} <: SimpleKernel{T}
    a::T
    κ::StandardKernel{T}
    function ScaledKernel(a::T, κ::StandardKernel{T})
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ)
    end
end
ScaledKernel{T<:FloatingPoint}(a::T, κ::StandardKernel{T}) = ScaledKernel{T}(a, κ)

function ScaledKernel{T}(a::Real, κ::StandardKernel{T})
    U = promote_type(typeof(a), T)
    ScaledKernel(convert(U, a), convert(Kernel{U}, κ))
end

for kernel_type in (:ScaledKernel, :SimpleKernel, :Kernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, ψ::ScaledKernel) 
            ScaledKernel(convert(T, ψ.a), convert(Kernel{T}, ψ.κ))
        end
    end
end

function kernel_function{T<:FloatingPoint}(ψ::ScaledKernel{T}, x::Vector{T}, y::Vector{T})
    ψ.a * kernel_function(ψ.κ, x, y)
end

function description_string{T<:FloatingPoint}(ψ::ScaledKernel{T})
    "ScaledKernel{$(T)}($(ψ.a)," * description_string(ψ.κ, false) * ")"
end
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

immutable KernelProduct{T<:FloatingPoint} <: CompositeKernel{T}
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

function KernelProduct{T,S}(a::Real, κ₁::StandardKernel{T}, κ₂::StandardKernel{S})
    U = promote_type(typeof(a), T, S)
    KernelProduct(convert(U, a), convert(Kernel{U}, κ₁), convert(Kernel{U}, κ₂))
end

for kernel_type in (:KernelProduct, :CompositeKernel, :Kernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, ψ::KernelProduct) 
            KernelProduct(convert(T, ψ.a), convert(Kernel{T}, ψ.κ₁),  convert(Kernel{T}, ψ.κ₂))
        end
    end
end

function kernel_function{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T})
    ψ.a * kernel_function(ψ.κ₁, x, y) * kernel_function(ψ.κ₂, x, y)
end

isposdef_kernel(ψ::KernelProduct) = isposdef_kernel(ψ.κ₁) & isposdef_kernel(ψ.κ₂)

function description_string{T<:FloatingPoint}(ψ::KernelProduct{T}) 
    "KernelProduct{$(T)}($(ψ.a)," * description_string(ψ.κ₁, false) * "," * (
    description_string(ψ.κ₂, false) * ")")
end

function show(io::IO, ψ::KernelProduct)
    print(io, description_string(ψ))
end

function *{T,S}(κ₁::StandardKernel{T}, κ₂::StandardKernel{S})
    KernelProduct(one(promote_type(T, S)), deepcopy(κ₁), deepcopy(κ₂))
end

*(κ::StandardKernel, ψ::ScaledKernel) = KernelProduct(ψ.a, deepcopy(κ), deepcopy(ψ.κ))
*(ψ::ScaledKernel, κ::StandardKernel) = *(κ, ψ)

*(ψ₁::ScaledKernel, ψ₂::ScaledKernel) = KernelProduct(ψ₁.a*ψ₂.a, deepcopy(ψ₁.κ), deepcopy(ψ₂.κ))

*(a::Real, ψ::KernelProduct) = KernelProduct(a * ψ.a, deepcopy(ψ.κ₁), deepcopy(ψ.κ₂))
*(ψ::KernelProduct, a::Real) = *(a, ψ)


#===========================================================================
  Kernel Sum
===========================================================================#

immutable KernelSum{T<:FloatingPoint} <: CompositeKernel{T}
    a₁::T
    κ₁::StandardKernel{T}
    a₂::T
    κ₂::StandardKernel{T}
    function KernelSum(a₁::T, κ₁::StandardKernel{T}, a₂::T, κ₂::StandardKernel{T})
        a₁ > 0 || error("a₁ = $(a₁) must be greater than zero.")
        a₂ > 0 || error("a₂ = $(a₂) must be greater than zero.")
        new(a₁, κ₁, a₂, κ₂)
    end
end
function KernelSum{T<:FloatingPoint}(a₁::T, κ₁::StandardKernel{T}, a₂::T, κ₂::StandardKernel{T})
    KernelSum{T}(a₁, κ₁, a₂, κ₂)
end

function KernelSum{T,S}(a₁::Real, κ₁::StandardKernel{T}, a₂::Real, κ₂::StandardKernel{S})
    U = promote_type(typeof(a₁), typeof(a₂), T, S)
    KernelSum{T}(convert(U, a₁), convert(Kernel{U}, κ₁), convert(U, a₂), convert(Kernel{U}, κ₂))
end

function kernel_function{T<:FloatingPoint}(ψ::KernelSum{T}, x::Vector{T}, y::Vector{T})
    ψ.a₁*kernel_function(ψ.κ₁, x, y) + ψ.a₂*kernel_function(ψ.κ₂, x, y)
end

isposdef_kernel(ψ::KernelSum) = isposdef_kernel(ψ.κ₁) | isposdef_kernel(ψ.κ₂)

function description_string{T<:FloatingPoint}(ψ::KernelSum{T}) 
    "KernelSum{$(T)}($(ψ.a₁)," * description_string(ψ.κ₁, false) * "," * "$(ψ.a₂)," * (
    description_string(ψ.κ₂, false) * ")")
end

function show(io::IO, ψ::KernelSum)
    print(io, description_string(ψ))
end

+(κ₁::StandardKernel, κ₂::StandardKernel) = KernelSum(1, deepcopy(κ₁), 1, deepcopy(κ₂))

+(κ::StandardKernel, ψ::ScaledKernel) = KernelSum(1, deepcopy(κ), ψ.a, deepcopy(ψ.κ))
+(ψ::ScaledKernel, κ::StandardKernel) = +(κ, ψ)

+(ψ₁::ScaledKernel, ψ₂::ScaledKernel) = KernelSum(ψ₁.a, deepcopy(ψ₁.κ), ψ₂.a, deepcopy(ψ₂.κ))

*(a::Real, ψ::KernelSum) = KernelSum(a*ψ.a₁, deepcopy(ψ.κ₁), a*ψ.a₂, deepcopy(ψ.κ₂))
*(ψ::KernelSum, a::Real) = *(a, ψ)
