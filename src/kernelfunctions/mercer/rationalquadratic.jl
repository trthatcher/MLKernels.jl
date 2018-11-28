# Abstract Rational-Quadratic Kernel =======================================================
abstract type AbstractRationalQuadraticKernel{T<:AbstractFloat} <: MercerKernel{T} end

@inline basefunction(::AbstractRationalQuadraticKernel) = SquaredEuclidean()


# Rational-Quadratic Kernel ================================================================
@doc raw"""
    RationalQuadraticKernel([α [,β]])

The rational quadratic kernel is an isotropic Mercer kernel given by the formula:

```math
\kappa(\mathbf{x},\mathbf{y})
= \left(1 +\alpha ||\mathbf{x} - \mathbf{y}||^{\gamma}\right)^{-\beta}
\qquad \alpha > 0, \; \beta > 0, \; 0 < \gamma \leq 1
```
where ``\alpha`` is a scaling parameter and ``\beta`` is a shape parameter.

The rational quadratic kernel is a special of the more general ``\gamma``-rational
quadratic kernel (see [`GammaRationalQuadraticKernel`](@ref)) with ``\gamma=1``.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> RationalQuadraticKernel()
RationalQuadraticKernel{Float64}(1.0,1.0)

julia> RationalQuadraticKernel(2.0f0)
RationalQuadraticKernel{Float32}(2.0,1.0)

julia> RationalQuadraticKernel(2.0f0, 2.0)
RationalQuadraticKernel{Float64}(2.0,2.0)
```
"""
struct RationalQuadraticKernel{T<:AbstractFloat} <: AbstractRationalQuadraticKernel{T}
    α::T
    β::T
    function RationalQuadraticKernel{T}(
            α::Real,
            β::Real
        ) where {T<:AbstractFloat}
        @check_args(RationalQuadraticKernel, α, α > zero(T), "α > 0")
        @check_args(RationalQuadraticKernel, β, β > zero(T), "β > 0")
        return new{T}(α, β)
    end
end
function RationalQuadraticKernel(
        α::T₁=1.0,
        β::T₂=one(floattype(T₁))
    ) where {T₁<:Real, T₂<:Real}
    RationalQuadraticKernel{floattype(T₁, T₂)}(α, β)
end

@inline function kappa(κ::RationalQuadraticKernel{T}, d²::T) where {T}
    return (one(T) + κ.α*d²)^(-κ.β)
end


# Gamma Rational-Quadratic Kernel ==========================================================
@doc raw"""
    GammaRationalKernel([α [,β [,γ]]])

The gamma-rational kernel is a generalization of the rational-quadratic kernel with an
additional shape parameter:

```math
\kappa(\mathbf{x},\mathbf{y})
= \left(1 +\alpha ||\mathbf{x} - \mathbf{y}||^{\gamma}\right)^{-\beta}
\qquad \alpha > 0, \; \beta > 0, \; 0 < \gamma \leq 1
```
where ``\alpha`` is a scaling parameter and ``\beta`` and ``\gamma`` are shape parameters.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> GammaRationalQuadraticKernel()
GammaRationalQuadraticKernel{Float64}(1.0,1.0,1.0)

julia> GammaRationalQuadraticKernel(2.0f0)
GammaRationalQuadraticKernel{Float32}(2.0,1.0,1.0)

julia> GammaRationalQuadraticKernel(2.0f0, 2.0f0)
GammaRationalQuadraticKernel{Float32}(2.0,2.0,1.0)

julia> GammaRationalQuadraticKernel(2.0f0, 2.0f0, 0.5f0)
GammaRationalQuadraticKernel{Float32}(2.0,2.0,0.5)
```
"""
struct GammaRationalQuadraticKernel{T<:AbstractFloat} <: AbstractRationalQuadraticKernel{T}
    α::T
    β::T
    γ::T
    function GammaRationalQuadraticKernel{T}(
            α::Real,
            β::Real,
            γ::Real
        ) where {T<:AbstractFloat}
        @check_args(GammaRationalQuadraticKernel, α, α > zero(T), "α > 0")
        @check_args(GammaRationalQuadraticKernel, β, β > zero(T), "β > 0")
        @check_args(GammaRationalQuadraticKernel, γ, one(T) >= γ > zero(T), "1 ⩾ γ > 0")
        return new{T}(α, β, γ)
    end
end
function GammaRationalQuadraticKernel(
        α::T₁ = 1.0,
        β::T₂ = one(floattype(T₁)),
        γ::T₃ = one(floattype(T₁, T₂))
    ) where {T₁<:Real, T₂<:Real, T₃<:Real}
    GammaRationalQuadraticKernel{floattype(T₁, T₂, T₃)}(α, β, γ)
end

@inline function kappa(κ::GammaRationalQuadraticKernel{T}, d²::T) where {T}
    return (one(T) + κ.α*(d²^κ.γ))^(-κ.β)
end