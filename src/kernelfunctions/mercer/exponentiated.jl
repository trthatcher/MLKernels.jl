@doc raw"""
    ExponentiatedKernel([α=1])

The exponentiated kernel is a Mercer kernel given by:

```
    κ(x,y) = exp(α⋅xᵀy)   α > 0
```
where `α` is a positive scaling parameter.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> ExponentiatedKernel()
ExponentiatedKernel{Float64}(1.0)

julia> ExponentiatedKernel(2)
ExponentiatedKernel{Float64}(2.0)

julia> ExponentiatedKernel(2.0f0)
ExponentiatedKernel{Float32}(2.0)
```
"""
struct ExponentiatedKernel{T<:AbstractFloat} <: MercerKernel{T}
    α::T
    function ExponentiatedKernel{T}(α::Real=T(1)) where {T<:AbstractFloat}
        @check_args(ExponentiatedKernel, α, α > zero(T), "α > 0")
        return new{T}(α)
    end
end
ExponentiatedKernel(α::T=1.0) where {T<:Real} = ExponentiatedKernel{promote_float(T)}(α)

@inline basefunction(::ExponentiatedKernel) = ScalarProduct()

@inline kappa(κ::ExponentiatedKernel{T}, xᵀy::T) where {T} = exp(κ.α*xᵀy)

function convert(
        ::Type{K},
        κ::ExponentiatedKernel
    ) where {K>:ExponentiatedKernel{T}} where T
    return ExponentiatedKernel{T}(κ.α)
end