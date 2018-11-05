@doc raw"""
    GammaExponentialKernel([α=1 [,γ=1]])

The gamma exponential kernel is a generalization of the exponential and squared exponential 
kernels:

```math
\kappa(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^{\gamma} 
\right) \qquad \alpha > 0, \; 0 < \gamma \leq 1
```
where ``\alpha`` is a scaling parameter and ``\gamma`` is a shape parameter.
"""
struct GammaExponentialKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaExponentialKernel{T}(α::Real, γ::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
function GammaExponentialKernel(α::T1=1.0, γ::T2=one(T1)) where {T1<:Real,T2<:Real}
    GammaExponentialKernel{floattype(T1,T2)}(α,γ)
end

@inline gammaexponentialkernel(z::T, α::T, γ::T) where {T<:AbstractFloat} = exp(-α*z^γ)

@inline basefunction(::GammaExponentialKernel) = SquaredEuclidean()
@inline function kappa(κ::GammaExponentialKernel{T}, z::T) where {T}
    gammaexponentialkernel(z, getvalue(κ.alpha), getvalue(κ.gamma))
end