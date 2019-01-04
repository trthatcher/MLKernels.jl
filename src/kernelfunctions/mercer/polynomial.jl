@doc raw"""
    PolynomialKernel([a=1 [,c=1 [,d=3]]])

The polynomial kernel is a Mercer kernel given by:

```
    κ(x,y) = (a⋅xᵀy + c)ᵈ   α > 0, c ≧ 0, d ∈ ℤ⁺
```

# Examples

```jldoctest; setup = :(using MLKernels)
julia> PolynomialKernel(2.0f0)
PolynomialKernel{Float32}(2.0,1.0,3)

julia> PolynomialKernel(2.0f0, 2.0)
PolynomialKernel{Float64}(2.0,2.0,3)

julia> PolynomialKernel(2.0f0, 2.0, 2)
PolynomialKernel{Float64}(2.0,2.0,2)
```
"""
struct PolynomialKernel{T<:AbstractFloat} <: MercerKernel{T}
    a::T
    c::T
    d::T
    function PolynomialKernel{T}(
            a::Real=T(1),
            c::Real=T(1),
            d::Real=T(3)
        ) where {T<:AbstractFloat}
        @check_args(PolynomialKernel, a, a >  zero(a), "a > 0")
        @check_args(PolynomialKernel, c, c >= zero(c), "c ≧ 0")
        @check_args(PolynomialKernel, d, d >= one(d) && d == trunc(d), "d ∈ ℤ₊")
        return new{T}(a, c, d)
    end
end

function PolynomialKernel(
        a::T₁=1.0,
        c::T₂=T₁(1),
        d::T₃=convert(promote_float(T₁,T₂), 3)
    ) where {T₁<:Real,T₂<:Real,T₃<:Real}
    T = promote_float(T₁,T₂,T₃)
    return PolynomialKernel{T}(a, c, d)
end

@inline basefunction(::PolynomialKernel) = ScalarProduct()

@inline function kappa(κ::PolynomialKernel{T}, xᵀy::T) where {T}
    return (κ.a*xᵀy + κ.c)^(κ.d)
end

function convert(::Type{K}, κ::PolynomialKernel) where {K>:PolynomialKernel{T}} where T
    return PolynomialKernel{T}(κ.a, κ.c, κ.d)
end