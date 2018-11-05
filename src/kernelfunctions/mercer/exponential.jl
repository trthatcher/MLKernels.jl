@doc raw"""
    ExponentialKernel([α=1])

The exponential kernel is given by the formula:

```math
\kappa(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||\right) 
\qquad \alpha > 0
```
  
where ``\alpha`` is a scaling parameter of the Euclidean distance. The exponential kernel, 
also known as the Laplacian kernel, is an isotropic Mercer kernel. The constructor is 
aliased by `LaplacianKernel`, so both names may be used:
"""
struct ExponentialKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    ExponentialKernel{T}(α::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
ExponentialKernel(α::T=1.0) where {T<:Real} = ExponentialKernel{floattype(T)}(α)
LaplacianKernel = ExponentialKernel

@inline exponentialkernel(z::T, α::T) where {T<:AbstractFloat} = exp(-α*sqrt(z))

@inline basefunction(::ExponentialKernel) = SquaredEuclidean()
@inline function kappa(κ::ExponentialKernel{T}, z::T) where {T<:AbstractFloat}
    exponentialkernel(z, getvalue(κ.alpha))
end