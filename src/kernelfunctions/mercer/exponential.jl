# Abstract Exponential Kernel ==============================================================

abstract type AbstractExponentialKernel{T<:AbstractFloat} <: MercerKernel{T} end

@inline basefunction(::AbstractExponentialKernel) = SquaredEuclidean()


# Exponential Kernel =======================================================================
@doc raw"""
    ExponentialKernel([α=1])

The exponential kernel is an isotropic Mercer kernel given by the formula:

```math
\kappa(\mathbf{x},\mathbf{y})
= \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||\right)
\qquad \alpha > 0
```
where ``\alpha`` is a scaling parameter.

This kernel may also be referred to as the exponential covariance function or the Laplacian
kernel (see [`LaplacianKernel`](@ref)). It is a special case of the more general
``\gamma``-exponential kernel with ``\gamma = 0.5`` (see [`GammaExponentialKernel`](@ref)).

# Examples

```jldoctest; setup = :(using MLKernels)
julia> ExponentialKernel()
ExponentialKernel{Float64}(1.0)

julia> ExponentialKernel(2.0f0)
ExponentialKernel{Float32}(2.0)
```
"""
struct ExponentialKernel{T<:AbstractFloat} <: AbstractExponentialKernel{T}
    α::T
    function ExponentialKernel{T}(α::Real) where {T<:AbstractFloat}
        @check_args(ExponentialKernel, α, α > zero(T), "α > 0")
        return new{T}(α)
    end
end
ExponentialKernel(α::T=1.0) where {T<:Real} = ExponentialKernel{floattype(T)}(α)

@inline kappa(κ::ExponentialKernel{T}, d²::T) where {T} = exp(-κ.α*√(d²))

"""
    LaplacianKernel([α=1])

Alias for [`ExponentialKernel`](@ref).
"""
const LaplacianKernel = ExponentialKernel


# Squared Exponential Kernel ===============================================================
@doc raw"""
    SquaredExponentialKernel([α=1])

The squared exponential kernel is an isotropic Mercer kernel given by the formula:

```math
\kappa(\mathbf{x},\mathbf{y})
= \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^{2}\right)
\qquad \alpha > 0
```
where ``\alpha`` is a scaling parameter.

This kernel may also be referred to as the squared exponential covariance function or the
Gaussian kernel (see [`GaussianKernel`](@ref)). In machine learning circles, it may be known
as the radial basis kernel (see RadialBasisKernel). It is a special case of the more general
``\gamma``-exponential kernel with ``\gamma = 1`` (see [`GammaExponentialKernel`](@ref)).

# Examples

```jldoctest; setup = :(using MLKernels)
julia> SquaredExponentialKernel()
SquaredExponentialKernel{Float64}(1.0)

julia> SquaredExponentialKernel(2.0f0)
SquaredExponentialKernel{Float32}(2.0)
```
"""
struct SquaredExponentialKernel{T<:AbstractFloat} <: AbstractExponentialKernel{T}
    α::T
    function SquaredExponentialKernel{T}(α::Real) where {T<:AbstractFloat}
        @check_args(SquaredExponentialKernel, α, α > zero(T), "α > 0")
        return new{T}(α)
    end
end
function SquaredExponentialKernel(α::T=1.0) where {T<:Real}
    return SquaredExponentialKernel{floattype(T)}(α)
end

@inline kappa(κ::SquaredExponentialKernel{T}, d²::T) where {T} = exp(-κ.α*d²)

"""
    GaussianKernel([α=1])

Alias for [`SquaredExponentialKernel`](@ref).
"""
const GaussianKernel = SquaredExponentialKernel

"""
    RadialBasisKernel([α=1])

Alias for [`SquaredExponentialKernel`](@ref).
"""
const RadialBasisKernel = SquaredExponentialKernel


# Gamma Exponential Kernel =================================================================
@doc raw"""
    GammaExponentialKernel([α=1 [,γ=1]])

The ``\gamma``-exponential kernel is an isotropic Mercer kernel given by the formula:

```math
\kappa(\mathbf{x},\mathbf{y})
= \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^{\gamma}\right)
\qquad \alpha > 0, \; 0 < \gamma \leq 1
```
where ``\alpha`` is a scaling parameter and ``\gamma`` is a shape parameter of the Euclidean
distance. There are two special cases that should be used if ``\gamma`` is a fixed
parameter:

  * When ``\gamma = 1``, use [`ExponentialKernel`](@ref)
  * When ``\gamma = 0.5``,  use [`SquaredExponentialKernel`](@ref)

# Examples

```jldoctest; setup = :(using MLKernels)
julia> GammaExponentialKernel()
GammaExponentialKernel{Float64}(1.0,1.0)

julia> GammaExponentialKernel(2.0f0)
GammaExponentialKernel{Float32}(2.0,1.0)

julia> GammaExponentialKernel(2.0, 0.5)
GammaExponentialKernel{Float64}(2.0,0.5)
```
"""
struct GammaExponentialKernel{T<:AbstractFloat} <: AbstractExponentialKernel{T}
    α::T
    γ::T
    function GammaExponentialKernel{T}(α::Real, γ::Real) where {T<:AbstractFloat}
        @check_args(GammaExponentialKernel, α, α > zero(T), "α > 0")
        @check_args(GammaExponentialKernel, γ, one(T) >= γ > zero(T), "1 ⩾ γ > 0")
        return new{T}(α, γ)
    end
end
function GammaExponentialKernel(α::T₁=1.0, γ::T₂=one(T₁)) where {T₁<:Real, T₂<:Real}
    return GammaExponentialKernel{floattype(T₁, T₂)}(α, γ)
end

@inline kappa(κ::GammaExponentialKernel{T}, d²::T) where {T} = exp(-κ.α*d²^κ.γ)