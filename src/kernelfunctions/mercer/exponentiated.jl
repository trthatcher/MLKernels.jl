@doc raw"""
    ExponentiatedKernel([a=1])

The exponentiated kernel is a Mercer kernel given by:

```math
\kappa(\mathbf{x},\mathbf{y}) = \exp\left(a \mathbf{x}^\intercal \mathbf{y} \right) 
\qquad a > 0
```
"""
struct ExponentiatedKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    ExponentiatedKernel{T}(α::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
ExponentiatedKernel(α::T1 = 1.0) where {T1<:Real} = ExponentiatedKernel{floattype(T1)}(α)

@inline exponentiatedkernel(z::T, α::T) where {T<:AbstractFloat} = exp(α*z)

@inline basefunction(::ExponentiatedKernel) = ScalarProduct()
@inline kappa(κ::ExponentiatedKernel{T}, z::T) where {T} = exponentiatedkernel(z, getvalue(κ.alpha))