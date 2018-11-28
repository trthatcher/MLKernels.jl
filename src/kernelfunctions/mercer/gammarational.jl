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
"""
struct RationalQuadraticKernel{Class, T<:AbstractFloat} <: MercerKernel{T}
    α::T
    β::T
    γ::T
    function RationalQuadraticKernel{Class, T}(
            α::Real,
            β::Real,
            γ::Real
        ) where {T<:AbstractFloat}
        @check_args(RationalQuadraticKernel, α, α > zero(T), "α > 0")
        @check_args(RationalQuadraticKernel, β, β > zero(T), "β > 0")
        @check_args(RationalQuadraticKernel, γ, one(T) >= γ > zero(T), "1 ⩾ γ > 0"
        return new{Class, T}(α, β, γ)
    end
end

function RationalQuadraticKernel(α::T1 = 1.0, β::T2 = one(T1)) where {T1<:Real,T2<:Real}
    RationalQuadraticKernel{:Standard, floattype(T1,T2)}(α, β, one(T1))
end

function RationalQuadraticKernel(α::T1, β::T2, γ::T3)) where {T1<:Real,T2<:Real,T3<:Real}
    RationalQuadraticKernel{floattype(T1,T2,T3)}(α,β,γ)
end

const GammaRationalQuadraticKernel{T} = RationalQuadraticKernel{:Gamma, T}

function GammaRationalQuadraticKernel(
        α::T1 = 1.0,
        β::T2 = one(T1),
        γ::T3 = one(floattype(T1,T2))
    ) where {T1<:Real,T2<:Real,T3<:Real}
    GammaRationalQuadraticKernel{floattype(T1,T2,T3)}(α,β,γ)
end

@inline basefunction(::GammaRationalKernel) = SquaredEuclidean()
@inline function kappa(κ::RationalQuadraticKernel{:Standard, T}, d²::T) where {T}
    return (one(T) + κ.α*d²)^(-κ.β)
end
@inline kappa(κ::RationalQuadraticKernel{:Gamma,    T}, d²::T) where {T} = (one(T) + κ.α*d²)^(-κ.β)
@inline kappa(κ::RationalQuadraticKernel{T}  , d²::T) where {T} = (one(T) + κ.α*(d²^κ.γ))^(-κ.β)