# Abstract Exponential Kernel ==============================================================

abstract type AbstractExponentialKernel{T<:AbstractFloat} <: MercerKernel{T} end

@inline basefunction(::AbstractExponentialKernel) = SquaredEuclidean()


# Exponential Kernel =======================================================================
@doc raw"""
    ExponentialKernel([α=1])

The exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(-α‖x-y‖)   α > 0
```

where `α` is a positive scaling parameter. See also [`SquaredExponentialKernel`](@ref) for
a related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> ExponentialKernel()
ExponentialKernel{Float64}(1.0)

julia> ExponentialKernel(2.0f0)
ExponentialKernel{Float32}(2.0)
```
"""
struct ExponentialKernel{T<:AbstractFloat} <: AbstractExponentialKernel{T}
    α::T
    function ExponentialKernel{T}(α::Real=T(1)) where {T<:AbstractFloat}
        @check_args(ExponentialKernel, α, α > zero(T), "α > 0")
        return new{T}(α)
    end
end
ExponentialKernel(α::T=1.0) where {T<:Real} = ExponentialKernel{promote_float(T)}(α)

@inline kappa(κ::ExponentialKernel{T}, d²::T) where {T} = exp(-κ.α*√(d²))

function convert(::Type{K}, κ::ExponentialKernel) where {K>:ExponentialKernel{T}} where T
    return ExponentialKernel{T}(κ.α)
end

"""
    LaplacianKernel([α=1])

Alias for [`ExponentialKernel`](@ref).
"""
const LaplacianKernel = ExponentialKernel


# Squared Exponential Kernel ===============================================================
@doc raw"""
    SquaredExponentialKernel([α=1])

The squared exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(-α‖x-y‖²)   α > 0
```

where `α` is a positive scaling parameter. See also [`ExponentialKernel`](@ref) for a
related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> SquaredExponentialKernel()
SquaredExponentialKernel{Float64}(1.0)

julia> SquaredExponentialKernel(2.0f0)
SquaredExponentialKernel{Float32}(2.0)
```
"""
struct SquaredExponentialKernel{T<:AbstractFloat} <: AbstractExponentialKernel{T}
    α::T
    function SquaredExponentialKernel{T}(α::Real=T(1)) where {T<:AbstractFloat}
        @check_args(SquaredExponentialKernel, α, α > zero(T), "α > 0")
        return new{T}(α)
    end
end
function SquaredExponentialKernel(α::T=1.0) where {T<:Real}
    return SquaredExponentialKernel{promote_float(T)}(α)
end

@inline kappa(κ::SquaredExponentialKernel{T}, d²::T) where {T} = exp(-κ.α*d²)

function convert(
        ::Type{K},
        κ::SquaredExponentialKernel
    ) where {K>:SquaredExponentialKernel{T}} where T
    return SquaredExponentialKernel{T}(κ.α)
end

"""
    GaussianKernel([α=1])

Alias of [`SquaredExponentialKernel`](@ref).
"""
const GaussianKernel = SquaredExponentialKernel

"""
    RadialBasisKernel([σ=1])

Create a [`SquaredExponentialKernel`](@ref) using the following
convention for [Radial Basis Function Kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel).

```
κ(x,y) = exp(-‖x-y‖²/σ²)
```
"""
RadialBasisKernel(σ) = SquaredExponentialKernel(1/2σ^2)


# Gamma Exponential Kernel =================================================================
@doc raw"""
    GammaExponentialKernel([α=1 [,γ=1]])

The ``\gamma``-exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(-α‖x-y‖²ᵞ)   α > 0, γ ∈ (0,1]
```
where `α` is a scaling parameter and `γ` is a shape parameter of the Euclidean distance.
When `γ = 1` use [`SquaredExponentialKernel`](@ref) and [`SquaredExponentialKernel`](@ref)
when `γ = 0.5` since these are more efficient implementations.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> GammaExponentialKernel()
GammaExponentialKernel{Float64}(1.0,1.0)

julia> GammaExponentialKernel(2.0f0)
GammaExponentialKernel{Float32}(2.0,1.0)

julia> GammaExponentialKernel(2.0, 0.5)
GammaExponentialKernel{Float64}(2.0,0.5)
```
"""
struct GammaExponentialKernel{T<:AbstractFloat} <: AbstractExponentialKernel{T}
    α::T
    γ::T
    function GammaExponentialKernel{T}(α::Real=T(1), γ::Real=T(1)) where {T<:AbstractFloat}
        @check_args(GammaExponentialKernel, α, α > zero(T), "α > 0")
        @check_args(GammaExponentialKernel, γ, one(T) >= γ > zero(T), "γ ∈ (0,1]")
        return new{T}(α, γ)
    end
end
function GammaExponentialKernel(α::T₁=1.0, γ::T₂=T₁(1)) where {T₁<:Real, T₂<:Real}
    return GammaExponentialKernel{promote_float(T₁,T₂)}(α, γ)
end

@inline kappa(κ::GammaExponentialKernel{T}, d²::T) where {T} = exp(-κ.α*d²^κ.γ)

function convert(
        ::Type{K},
        κ::GammaExponentialKernel
    ) where {K>:GammaExponentialKernel{T}} where T
    return GammaExponentialKernel{T}(κ.α, κ.γ)
end
