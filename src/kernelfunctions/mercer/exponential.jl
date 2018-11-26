@doc raw"""
    ExponentialKernel([α=1 [,γ=1]])

The exponential kernel is an isotropic Mercer kernel given by the formula:

```math
\kappa(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^{\gamma}
\right) \qquad \alpha > 0, \; 0 < \gamma \leq 1
```
where ``\alpha`` is a scaling parameter and ``\gamma`` is a shape parameter of the Euclidean
distance.

When ``\gamma = 1``, the kernel simplifies to a function of the squared Euclidean distance.
In this case, the kernel is commonly known as the squared exponential covariance function or
the Gaussian kernel.

When ``\gamma = 0.5``, the kernel simplifies to a function of the Euclidean distance. In
this case, the kernel may be referred to as the exponential covariance function or the
Laplacian kernel.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> ExponentialKernel()
ExponentialKernel{Float64}(1.0,0.5)

julia> ExponentialKernel(2.0f0)
ExponentialKernel{Float32}(2.0,0.5)

julia> ExponentialKernel(2.0, 0.5)
ExponentialKernel{Float64}(2.0,0.5)
```
"""
struct ExponentialKernel{Class, T<:AbstractFloat} <: MercerKernel{T}
    α::T
    γ::T
    function ExponentialKernel{Class, T}(α::Real, γ::Real) where {Class, T<:AbstractFloat}
        @check_args(ExponentialKernel, α, α > zero(T), "α > 0")
        @check_args(ExponentialKernel, γ, one(T) >= γ > zero(T), "1 ⩾ γ > 0")
        return new{Class, T}(α, γ)
    end
end

function ExponentialKernel(α::T1, γ::T2) where {T1<:Real, T2<:Real}
    ExponentialKernel{:Gamma, floattype(T1,T2)}(α, γ)
end

function ExponentialKernel(α::T=1.0) where {T<:Real}
    ExponentialKernel{:Standard, floattype(T)}(α, convert(T, 0.5))
end

@doc raw"""
    LaplacianKernel([α=1])

The Laplacian kernel is a special case of the exponential kernel, but with the parameter
``\gamma = 0.5``. See [`ExponentialKernel`](@ref).

# Examples

```jldoctest; setup = :(using MLKernels)
julia> ExponentialKernel()
ExponentialKernel{Float64}(1.0,0.5)

julia> LaplacianKernel(2.0f0)
ExponentialKernel{Float32}(2.0,0.5)
```
"""
const LaplacianKernel{T} = ExponentialKernel{:Standard, T}

function LaplacianKernel(α::T=1.0) where {T<:Real}
    ExponentialKernel{:Standard, floattype(T)}(α, convert(T, 0.5))
end

@doc raw"""
    SquaredExponentialKernel([α=1])

The squared exponential kernel is a special case of the exponential kernel, but with the
parameter ``\gamma = 1``. See [`ExponentialKernel`](@ref).

# Examples

```jldoctest; setup = :(using MLKernels)
julia> SquaredExponentialKernel()
ExponentialKernel{Float64}(1.0,1.0)

julia> SquaredExponentialKernel(2.0f0)
ExponentialKernel{Float32}(2.0,1.0)
```
"""
const SquaredExponentialKernel{T} = ExponentialKernel{:Squared, T}

"""
    GaussianKernel([α=1])

See [`SquaredExponentialKernel`](@ref).
"""
const GaussianKernel    = SquaredExponentialKernel

"""
    RadialBasisKernel([α=1])

See [`SquaredExponentialKernel`](@ref).
"""
const RadialBasisKernel = SquaredExponentialKernel

function SquaredExponentialKernel(α::T=1.0) where {T<:Real}
    ExponentialKernel{:Squared, floattype(T)}(α, one(T))
end

@doc raw"""
    GammaExponentialKernel([α=1 [,γ=1]])

See [`ExponentialKernel`](@ref).
"""
const GammaExponentialKernel{T} = ExponentialKernel{:Gamma, T}

function GammaExponentialKernel(α::T1=1.0, γ::T2=one(T1)) where {T1<:Real,T2<:Real}
    ExponentialKernel{:Gamma, floattype(T1,T2)}(α, γ)
end

@inline basefunction(::GammaExponentialKernel) = SquaredEuclidean()
@inline kappa(κ::ExponentialKernel{:Squared, T}, d²::T) where {T} = exp(-κ.α*d²)
@inline kappa(κ::ExponentialKernel{:Standard,T}, d²::T) where {T} = exp(-κ.α*√(d²))
@inline kappa(κ::ExponentialKernel{:Gamma,   T}, d²::T) where {T} = exp(-κ.α*d²^κ.γ)