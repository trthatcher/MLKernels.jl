#===================================================================================================
  Mercer Kernel Type & Related Functions
===================================================================================================#

abstract Kernel

abstract SimpleKernel{T<:FloatingPoint} <: Kernel
abstract CompositeKernel{T<:FloatingPoint} <: Kernel

abstract TransformedKernel{T<:FloatingPoint} <: SimpleKernel{T}
abstract StandardKernel{T<:FloatingPoint} <: SimpleKernel{T}

typealias ScalableKernel{T<:FloatingPoint} Union(StandardKernel{T}, TransformedKernel{T})

call(κ::Kernel, args...) = kernel_function(κ, args...)

isposdef_kernel(κ::Kernel) = false
is_euclidean_distance(κ::Kernel) = false
is_scalar_product(κ::Kernel) = false


#===================================================================================================
  Transformed and Scaled Mercer Kernels
===================================================================================================#

#== Exponential Mercer Kernel ====================#

immutable ExponentialKernel{T<:FloatingPoint} <: TransformedKernel{T}
    κ::StandardKernel{T}
    ExponentialKernel(κ::StandardKernel{T}) = new(κ)
end
ExponentialKernel{T<:FloatingPoint}(κ::StandardKernel{T}) = ExponentialKernel{T}(κ)

@inline function scalar_kernel_function{T<:FloatingPoint}(ψ::ExponentialKernel{T}, x::T)
    exp(scalar_kernel_function(ψ.κ, x))
end

function scalar_kernel_function!{T<:FloatingPoint}(ψ::ExponentialKernel{T}, G::Array{T})
    scalar_kernel_function!(ψ.κ, G)
    @inbounds for i = 1:length(G)
        G[i] = exp(G[i])
    end
    G
end

function scalar_kernel_function{T<:FloatingPoint}(ψ::ExponentialKernel{T}, G::Array{T})
    scalar_kernel_function!(ψ, copy(G))
end

@inline function kernel_function{T<:FloatingPoint}(ψ::ExponentialKernel{T}, x::Vector{T}, 
                                                   y::Vector{T})
    exp(kernel_function(ψ.κ, x, y))
end

isposdef_kernel(ψ::ExponentialKernel) = isposdef_kernel(ψ.κ)
is_euclidean_distance(κ::ExponentialKernel) = is_euclidean_distance(ψ.κ)
is_scalar_product(κ::ExponentialKernel) = is_scalar_product(ψ.κ)

function description_string{T<:FloatingPoint}(ψ::ExponentialKernel{T})
    "ExponentialKernel{$(T)}(" * description_string(ψ.κ) * ")"
end

function show(io::IO, ψ::ExponentialKernel)
    print(io, " " * description_string(ψ))
end

exp(κ::StandardKernel) = ExponentialKernel(deepcopy(κ))


#== Exponentiated Mercer Kernel ====================#

immutable ExponentiatedKernel{T<:FloatingPoint} <: TransformedKernel{T}
    κ::StandardKernel{T}
    a::T
    function ExponentiatedKernel(κ::StandardKernel{T}, a::T)
        a > 0 || error("a = $(a) must be a non-negative number.")
        new(κ, a)
    end
end
function ExponentiatedKernel{T<:FloatingPoint}(κ::StandardKernel{T}, a::T)
    ExponentiatedKernel{T}(κ, a)
end

function scalar_kernel_function!{T<:FloatingPoint}(ψ::ExponentiatedKernel{T}, G::Array{T})
    scalar_kernel_function!(ψ.κ, G)
    for i = 1:length(G)
        G[i] = G[i] ^ ψ.a
    end
    G
end

function scalar_kernel_function{T<:FloatingPoint}(ψ::ExponentiatedKernel{T}, G::Array{T})
    scalar_kernel_function!(ψ, copy(G))
end

@inline function kernel_function{T<:FloatingPoint}(ψ::ExponentiatedKernel{T}, x::Vector{T}, 
                                                   y::Vector{T})
    kernel_function(ψ.κ, x, y) ^ ψ.a
end

isposdef_kernel(ψ::ExponentiatedKernel) = isposdef_kernel(ψ.κ)
is_euclidean_distance(κ::ExponentiatedKernel) = is_euclidean_distance(ψ.κ)
is_scalar_product(κ::ExponentiatedKernel) = is_scalar_product(ψ.κ)

function description_string{T<:FloatingPoint}(ψ::ExponentiatedKernel{T})
    "ExponentiatedKernel{$(T)}(" * description_string(ψ.κ) * ", $(ψ.a))"
end

function show(io::IO, ψ::ExponentiatedKernel)
    print(io, " " * description_string(ψ))
end

^{T<:FloatingPoint}(κ::StandardKernel{T}, a::T) = ExponentiatedKernel(deepcopy(κ), a)


#== Scaled Mercer Kernel ====================#

type ScaledKernel{T<:FloatingPoint} <: SimpleKernel{T}
    a::T
    κ::ScalableKernel{T}
    function ScaledKernel(a::T, κ::ScalableKernel{T})
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ)
    end
end
ScaledKernel{T<:FloatingPoint}(a::T, κ::ScalableKernel{T}) = ScaledKernel{T}(a, κ)

@inline function kernel_function(ψ::ScaledKernel)
    if ψ.a == 1
        k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernel_function(ψ.κ)(x, y)
    end
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernel_function(ψ.κ)(x, y) * convert(T, ψ.a)
end

description_string(ψ::ScaledKernel) = "$(ψ.a) * " * description_string(ψ.κ)
isposdef_kernel(ψ::ScaledKernel) = isposdef_kernel(ψ.κ)

function show(io::IO, ψ::ScaledKernel)
    println(io, "Scaled Mercer Kernel:")
    print(io, " " * description_string(ψ))
end

*(a::Real, κ::ScalableKernel) = ScaledKernel(a, deepcopy(κ))
*(κ::ScalableKernel, a::Real) = *(a, κ)

*(a::Real, ψ::ScaledKernel) = ScaledKernel(a * ψ.a, deepcopy(ψ.κ))
*(ψ::ScaledKernel, a::Real) = *(a, ψ)


#===================================================================================================
  Composite Kernels
===================================================================================================#

#== Mercer Kernel Product ====================#

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


#== Mercer Kernel Sum ====================#

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


