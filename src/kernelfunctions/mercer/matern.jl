@doc raw"""
    MaternKernel([ν=1 [, θ=1]])

The Matern kernel is a Mercer kernel with parameters `ν > 0` and `ρ > 0`. See the published
documentation for the full definition of the function.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> MaternKernel()
MaternKernel{Float64}(1.0,1.0)

julia> MaternKernel(2.0f0)
MaternKernel{Float32}(2.0,1.0)

julia> MaternKernel(2.0f0, 2.0)
MaternKernel{Float64}(2.0,2.0)
```
"""
struct MaternKernel{T<:AbstractFloat} <: MercerKernel{T}
    ν::T
    ρ::T
    function MaternKernel{T}(ν::Real=T(1), ρ::Real=T(1)) where {T<:AbstractFloat}
        @check_args(MaternKernel, ν, ν > zero(T), "ν > 0")
        @check_args(MaternKernel, ρ, ρ > zero(T), "ρ > 0")
        return new{T}(ν, ρ)
    end
end
function MaternKernel(ν::T₁=1.0, ρ::T₂=T₁(1)) where {T₁<:Real,T₂<:Real}
    MaternKernel{promote_float(T₁,T₂)}(ν,ρ)
end

@inline basefunction(::MaternKernel) = SquaredEuclidean()

@inline function kappa(κ::MaternKernel{T}, d²::T) where {T}
    d = √(d²)
    d = d < eps(T) ? eps(T) : d  # If d is zero, besselk will return NaN
    tmp = √(2κ.ν)*d/κ.ρ
    return (convert(T, 2)^(one(T) - κ.ν))*(tmp^κ.ν)*besselk(κ.ν, tmp)/gamma(κ.ν)
end

function convert(::Type{K}, κ::MaternKernel) where {K>:MaternKernel{T}} where T
    return MaternKernel{T}(κ.ν, κ.ρ)
end