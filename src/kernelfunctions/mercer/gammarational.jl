@doc raw"""
    GammaRationalKernel([α [,β [,γ]]])
  
The gamma-rational kernel is a generalization of the rational-quadratic kernel with an 
additional shape parameter:

```math
\kappa(\mathbf{x},\mathbf{y})
= \left(1 +\alpha ||\mathbf{x},\mathbf{y}||^{\gamma}\right)^{-\beta} 
\qquad \alpha > 0, \; \beta > 0, \; 0 < \gamma \leq 1
```

where ``\alpha`` is a scaling parameter and ``\beta`` and ``\gamma`` are shape parameters.
"""
struct GammaRationalKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaRationalKernel{T}(α::Real, β::Real, γ::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,β), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
function GammaRationalKernel(
        α::T1 = 1.0,
        β::T2 = one(T1),
        γ::T3 = one(floattype(T1,T2))
    ) where {T1<:Real,T2<:Real,T3<:Real}
    GammaRationalKernel{floattype(T1,T2,T3)}(α,β,γ)
end

@inline gammarationalkernel(z::T, α::T, β::T, γ::T) where {T<:AbstractFloat} = (1 + α*(z^γ))^(-β)

@inline basefunction(::GammaRationalKernel) = SquaredEuclidean()
@inline function kappa(κ::GammaRationalKernel{T}, z::T) where {T}
    gammarationalkernel(z, getvalue(κ.alpha), getvalue(κ.beta), getvalue(κ.gamma))
end