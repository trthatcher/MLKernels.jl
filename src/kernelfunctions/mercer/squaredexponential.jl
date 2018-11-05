@doc raw"""
    SquaredExponentialKernel([α=1])

The squared exponential kernel, or alternatively the Gaussian kernel, is identical to the 
exponential kernel except that the Euclidean distance is squared:

```math
\kappa(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^2\right) 
\qquad \alpha > 0
```

where ``\alpha`` is a scaling parameter of the squared Euclidean distance.
Just like the exponential kernel, the squared exponential kernel is an
isotropic Mercer kernel. The squared exponential kernel is more commonly known
as the radial basis kernel within machine learning communities.
"""
struct SquaredExponentialKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    SquaredExponentialKernel{T}(α::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
SquaredExponentialKernel(α::T=1.0) where {T<:Real} = SquaredExponentialKernel{floattype(T)}(α)
GaussianKernel = SquaredExponentialKernel
RadialBasisKernel = SquaredExponentialKernel

@inline squaredexponentialkernel(z::T, α::T) where {T<:AbstractFloat} = exp(-α*z)

@inline basefunction(::SquaredExponentialKernel) = SquaredEuclidean()
@inline function kappa(κ::SquaredExponentialKernel{T}, z::T) where {T}
    squaredexponentialkernel(z, getvalue(κ.alpha))
end