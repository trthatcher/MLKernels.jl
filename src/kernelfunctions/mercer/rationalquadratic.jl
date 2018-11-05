@doc raw"""
    RationalQuadraticKernel([α=1 [,β=1]])

The rational-quadratic kernel is given by:

```math
\kappa(\mathbf{x},\mathbf{y}) 
= \left(1 +\alpha ||\mathbf{x},\mathbf{y}||^2\right)^{-\beta} 
\qquad \alpha > 0, \; \beta > 0
```

where ``\alpha`` is a scaling parameter and ``\beta`` is a shape parameter. This kernel can 
be seen as an infinite sum of Gaussian kernels. If one sets ``\alpha = \alpha_0 / \beta``, 
then taking the limit ``\beta \rightarrow \infty`` results in the Gaussian kernel with 
scaling parameter ``\alpha_0``.
"""
struct RationalQuadraticKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    RationalQuadraticKernel{T}(α::Real, β::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,β), interval(OpenBound(zero(T)), nothing))
    )
end
function RationalQuadraticKernel(α::T1 = 1.0, β::T2 = one(T1)) where {T1<:Real,T2<:Real}
    RationalQuadraticKernel{floattype(T1,T2)}(α, β)
end

@inline rationalquadratickernel(z::T, α::T, β::T) where {T<:AbstractFloat} = (1 + α*z)^(-β)

@inline basefunction(::RationalQuadraticKernel) = SquaredEuclidean()
@inline function kappa(κ::RationalQuadraticKernel{T}, z::T) where {T}
    rationalquadratickernel(z, getvalue(κ.alpha), getvalue(κ.beta))
end