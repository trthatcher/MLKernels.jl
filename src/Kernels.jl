#===================================================================================================
  Mercer Kernel Type & Related Functions
===================================================================================================#

abstract Kernel

abstract SimpleKernel <: Kernel
abstract CompositeKernel <: Kernel

abstract StandardKernel <: SimpleKernel
abstract TransformedKernel <: SimpleKernel

ScalableKernel = Union(StandardKernel, TransformedKernel)

call{T<:FloatingPoint}(κ::Kernel, x::Array{T}, y::Array{T}) = kernel_function(κ)(x, y)

isposdef_kernel(κ::Kernel) = false
isposdef(κ::Kernel) = isposdef_kernel(κ)

is_euclidean_distance(κ::Kernel) = false
is_scalar_product(κ::Kernel) = false


#===================================================================================================
  Auxiliary Functions
===================================================================================================#

# xᵗy
@inline function scalar_product{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    n = length(x)
    BLAS.dot(n, x, 1, y, 1)
end

# (x-y)ᵗ(x-y) = ϵᵗϵ
@inline function euclidean_distance{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    n = length(x)
    ϵ = BLAS.axpy!(n, -one(T), y, 1, copy(x), 1)
    BLAS.dot(n, ϵ, 1, ϵ, 1)
end


#===================================================================================================
  Transformed and Scaled Mercer Kernels
===================================================================================================#

#== Exponential Mercer Kernel ====================#

type ExponentialKernel <: TransformedKernel
    κ::StandardKernel
    ExponentialKernel(κ::StandardKernel) = new(κ)
end

@inline kernel_function(ψ::ExponentialKernel) = exp(kernel_function(ψ.κ)(x, y))
isposdef_kernel(ψ::ExponentialKernel) = isposdef_kernel(ψ.κ)


description_string(ψ::ExponentialKernel) = "exp(" * description_string(ψ.κ) * ")"

function show(io::IO, ψ::ExponentialKernel)
    println(io, "Exponential Mercer Kernel:")
    print(io, " " * description_string(ψ))
end

exp(κ::StandardKernel) = ExponentialKernel(deepcopy(κ))


#== Exponentiated Mercer Kernel ====================#

type ExponentiatedKernel <: TransformedKernel
    κ::StandardKernel
    a::Integer
    function ExponentiatedKernel(κ::StandardKernel, a::Real)
        a > 0 || error("a = $(a) must be a non-negative number.")
        new(κ, a)
    end
end

@inline kernel_function(ψ::ExponentiatedKernel) = (kernel_function(ψ.κ)(x, y)) ^ ψ.a
isposdef_kernel(ψ::ExponentiatedKernel) = isposdef_kernel(ψ.κ)

description_string(ψ::ExponentiatedKernel) = description_string(ψ.κ) * " ^ $(ψ.a)"

function show(io::IO, ψ::ExponentiatedKernel)
    println(io, "Exponentiated Mercer Kernel:")
    print(io, " " * description_string(ψ))
end

^(κ::StandardKernel, a::Integer) = ExponentiatedKernel(deepcopy(κ), a)


#== Scaled Mercer Kernel ====================#

type ScaledKernel <: SimpleKernel
    a::Real
    κ::ScalableKernel
    function ScaledKernel(a::Real, κ::ScalableKernel)
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ)
    end
end

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

#===================================================================================================
  Standard Kernels
===================================================================================================#

include("standardkernels.jl")

#=================================================
  Generic Kernels
=================================================#

#== Pointwise Product Kernel ==#

@inline function pointwiseproductkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, f::Function)
    f(x) * f(y)
end

type PointwiseProductKernel <: StandardKernel
    f::Function
    posdef::Bool
    function PointwiseProductKernel(f::Function, posdef::Bool = false)
        method_exists(f, (Array{Float32},)) && method_exists(f, (Array{Float64},)) || (
            error("f = $(f) must map f: ℝⁿ → ℝ (define methods for both Array{Float32} and " * ( 
                  "Array{Float64}).")))
        new(f, posdef)
    end
end

arguments(κ::PointwiseProductKernel) = (κ.f, κ.posdef)
isposdef_kernel(κ::PointwiseProductKernel) = κ.posdef

@inline function kernel_function(κ::PointwiseProductKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = pointwiseproductkernel(x, y, copy(κ.f))
end

formula_string(κ::PointwiseProductKernel) = "k(x,y) = f(x)f(y)"
argument_string(κ::PointwiseProductKernel) = "f = $(κ.f)"
description_string(κ::PointwiseProductKernel) = "PointwiseProductKernel(f=$(κ.f))"

function description(κ::PointwiseProductKernel)
    print(
        """ 
         Pointwise Product Kernel:
         ===================================================================
         The pointwise product kernel is the product of a real-valued multi-
         variate function applied to each of the vector arguments:

             k(x,y) = f(x)f(y)    x ∈ ℝⁿ, y ∈ ℝⁿ, f: ℝⁿ → ℝ
        """
    )
end


#== Generic Kernel ====================#

type GenericKernel <: StandardKernel
    k::Function
    posdef::Bool
    function GenericKernel(k::Function, posdef::Bool = false)
        method_exists(f, (Array{Float32}, Array{Float32})) && (
            method_exists(f, (Array{Float64}, Array{Float64})) || (
            error("k = $(f) must map k: ℝⁿ×ℝⁿ → ℝ (define methods for both" * (
                  "Array{Float32} and Array{Float64})."))))
        new(k, posdef)
    end
end

arguments(κ::GenericKernel) = (κ.k,)
isposdef_kernel(κ::GenericKernel) = κ.posdef

kernel_function(κ::GenericKernel) = copy(κ.k)

formula_string(κ::GenericKernel) = "k(x,y)"
argument_string(κ::GenericKernel) = "k = $(κ.k)"
description_string(κ::GenericKernel) = "GenericKernel(k=$(κ.k))"

function description(κ::GenericKernel)
    print(
        """ 
         Generic Kernel:
         ===================================================================
         Customized definition:

             k: ℝⁿ×ℝⁿ → ℝ
        """
    )
end
