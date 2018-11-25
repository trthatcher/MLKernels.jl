@doc raw"""
    GammaExponentialKernel([α=1 [,γ=1]])

The gamma exponential kernel is a generalization of the exponential and squared exponential
kernels:

```math
\kappa(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^{\gamma}
\right) \qquad \alpha > 0, \; 0 < \gamma \leq 1
```
where ``\alpha`` is a scaling parameter and ``\gamma`` is a shape parameter.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> GammaExponentialKernel()
GammaExponentialKernel{Float64}(1.0,1.0)

julia> GammaExponentialKernel(2.0f0)
GammaExponentialKernel{Float32}(2.0,1.0)

julia> GammaExponentialKernel(2.0f0, 0.50)
GammaExponentialKernel{Float64}(2.0,0.5)
```
"""
struct GammaExponentialKernel{T<:AbstractFloat,Θ} <: MercerKernel{T}
    α::T
    γ::T
    function GammaExponentialKernel{T}(α::Real, γ::Real) where {T<:AbstractFloat}
        @check_args(GammaExponentialKernel, α, α > zero(T), "α > 0")
        @check_args(GammaExponentialKernel, γ, one(T) >= γ > zero(T), "1 ⩾ γ > 0")
        if γ == one(T)
            return new{T,1}(α, γ)
        elseif γ == convert(T, 0.5)
            return new{T,2}(α, γ)
        else
            return new{T,0}(α, γ)
        end
    end
end
function GammaExponentialKernel(α::T1=1.0, γ::T2=one(T1)) where {T1<:Real,T2<:Real}
    GammaExponentialKernel{floattype(T1,T2)}(α, γ)
end

@inline basefunction(::GammaExponentialKernel) = SquaredEuclidean()
@inline kappa(κ::GammaExponentialKernel{T,1}, d²::T) where {T} = exp(-κ.α*d²)
@inline kappa(κ::GammaExponentialKernel{T,2}, d²::T) where {T} = exp(-κ.α*√(d²))
@inline kappa(κ::GammaExponentialKernel{T}  , d²::T) where {T} = exp(-κ.α*d²^κ.γ)