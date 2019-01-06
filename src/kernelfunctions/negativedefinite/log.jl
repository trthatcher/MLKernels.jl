@doc raw"""
    LogKernel([α [,γ]])

The Log Kernel is a negative definite kernel given by the formula:

```
    κ(x,y) = log(1 + α‖x-y‖²ᵞ)   α > 0, γ ∈ (0,1]
```
where `α` is a scaling parameter and `γ` is a shape parameter of the Euclidean distance.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> LogKernel()
LogKernel{Float64}(1.0,1.0)

julia> LogKernel(0.5f0)
LogKernel{Float32}(0.5,1.0)

julia> LogKernel(0.5, 0.5)
LogKernel{Float64}(0.5,0.5)
```
"""
struct LogKernel{T<:AbstractFloat} <: NegativeDefiniteKernel{T}
    α::T
    γ::T
    function LogKernel{T}(α::Real=T(1), γ::Real=T(1)) where {T<:AbstractFloat}
        @check_args(LogKernel, α, α > zero(T), "α > 0")
        @check_args(LogKernel, γ, one(T) >= γ > zero(T), "γ ∈ (0,1]")
        return new{T}(α, γ)
    end
end
function LogKernel(α::T₁=1.0, γ::T₂=T₁(1)) where {T₁<:Real,T₂<:Real}
    LogKernel{promote_float(T₁,T₂)}(α, γ)
end

@inline basefunction(::LogKernel) = SquaredEuclidean()

@inline kappa(κ::LogKernel{T}, d²::T) where {T} = log(one(T) + κ.α*d²^(κ.γ))

function convert(::Type{K}, κ::LogKernel) where {K>:LogKernel{T}} where T
    return LogKernel{T}(κ.α, κ.γ)
end