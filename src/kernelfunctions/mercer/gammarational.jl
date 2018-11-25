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
struct GammaRationalKernel{T<:AbstractFloat,θ} <: MercerKernel{T}
    α::T
    β::T
    γ::T
    function GammaRationalKernel{T}(α::Real, β::Real, γ::Real) where {T<:AbstractFloat}
        @check_args(GammaRationalKernel, α, α > zero(T), "α > 0")
        @check_args(GammaRationalKernel, β, β > zero(T), "β > 0")
        @check_args(GammaRationalKernel, γ, one(T) >= γ > zero(T), "1 ⩾ γ > 0"
        θ = 0
        if γ == one(T)
            θ = β == one(T) ? 1 : 2
        elseif γ == convert(T, 0,5)
            θ = β == one(T) ? 3 : 4
        elseif β == one(T)
            θ = 5
        end
        return new{T,θ}(α, β, γ)
    end
end
function GammaRationalKernel(
        α::T1 = 1.0,
        β::T2 = one(T1),
        γ::T3 = one(floattype(T1,T2))
    ) where {T1<:Real,T2<:Real,T3<:Real}
    GammaRationalKernel{floattype(T1,T2,T3)}(α,β,γ)
end

@inline basefunction(::GammaRationalKernel) = SquaredEuclidean()
@inline kappa(κ::GammaRationalKernel{T,1}, d²::T) where {T} = one(T)/(one(T) + κ.α*d²)
@inline kappa(κ::GammaRationalKernel{T}  , d²::T) where {T} = (one(T) + κ.α*(d²^κ.γ))^(-κ.β)