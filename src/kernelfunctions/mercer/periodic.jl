@doc raw"""
    PeriodicKernel([α=1])

The periodic kernel is a mercer kernel with parameter `α > 0`. See the published
documentation for the full definition of the function.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> PeriodicKernel()
PeriodicKernel{Float64}(1.0)

julia> PeriodicKernel(2.0f0)
PeriodicKernel{Float32}(2.0)
```
"""
struct PeriodicKernel{T<:AbstractFloat} <: MercerKernel{T}
    α::T
    function PeriodicKernel{T}(α::Real) where {T<:AbstractFloat}
        @check_args(PeriodicKernel, α, α > zero(α), "α > 0")
        new{T}(α)
    end
end
PeriodicKernel(α::T₁ = 1.0) where {T₁<:Real} = PeriodicKernel{floattype(T₁)}(α)

@inline basefunction(::PeriodicKernel) = SineSquared()

@inline function kappa(κ::PeriodicKernel{T}, z::T) where {T}
    return exp(-κ.α*z)
end