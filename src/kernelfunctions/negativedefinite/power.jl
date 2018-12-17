@doc raw"""
    PowerKernel([γ=1])

The Power Kernel is a negative definite kernel given by:
```
    κ(x,y) = ‖x-y‖²ᵞ   0 < γ ≦ 1
```
where `γ` is a shape parameter of the Euclidean distance.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> PowerKernel()
PowerKernel{Float64}(1.0)

julia> PowerKernel(0.5f0)
PowerKernel{Float32}(0.5)
```
"""
struct PowerKernel{T<:AbstractFloat} <: NegativeDefiniteKernel{T}
    γ::T
    function PowerKernel{T}(γ::Real) where {T<:AbstractFloat}
        @check_args(PowerKernel, γ, one(T) >= γ > zero(T), "1 ⩾ γ > 0")
        new{T}(γ)
    end
end
PowerKernel(γ::T = 1.0) where {T<:Real} = PowerKernel{floattype(T)}(γ)

@inline basefunction(::PowerKernel) = SquaredEuclidean()

@inline kappa(κ::PowerKernel{T}, d²::T) where {T} = d²^κ.γ