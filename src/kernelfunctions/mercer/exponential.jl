@doc raw"""
    ExponentialKernel([α=1])

The exponential kernel is given by the formula:

```math
\kappa(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||\right)
\qquad \alpha > 0
```

where ``\alpha`` is a scaling parameter of the Euclidean distance. The exponential kernel,
also known as the Laplacian kernel, is an isotropic Mercer kernel. The constructor is
aliased by `LaplacianKernel`, so both names may be used.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> ExponentialKernel()
ExponentialKernel{Float64}(1.0)

julia> ExponentialKernel(2)
ExponentialKernel{Float64}(2.0)

julia> ExponentialKernel(2.0f0)
ExponentialKernel{Float32}(2.0)
```
"""
struct ExponentialKernel{T<:AbstractFloat} <: MercerKernel{T}
    α::T
    function ExponentialKernel{T}(α::Real) where {T<:AbstractFloat}
        @check_args(ExponentialKernel, α, α > zero(T), "α > 0")
        return new{T}(α)
    end
end
ExponentialKernel(α::T=1.0) where {T<:Real} = ExponentialKernel{floattype(T)}(α)
LaplacianKernel = ExponentialKernel

@inline basefunction(::ExponentialKernel) = SquaredEuclidean()
@inline kappa(κ::ExponentialKernel{T}, d²::T) where {T} = exp(-κ.α*√(d²))