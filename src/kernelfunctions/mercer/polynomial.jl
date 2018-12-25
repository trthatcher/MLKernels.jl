@doc raw"""
    PolynomialKernel([a=1 [,c=1 [,d=3]]])

The polynomial kernel is a Mercer kernel given by:

```
    κ(x,y) = (αxᵀy + c)ᵈ   α > 0, c ≧ 0, d ∈ ℤ⁺
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
struct PolynomialKernel{T<:AbstractFloat,U<:Integer} <: MercerKernel{T}
    a::T
    c::T
    d::U
    function PolynomialKernel{T,U}(
            a::Real,
            c::Real,
            d::Real
        ) where {T<:AbstractFloat,U<:Integer}
        @check_args(PolynomialKernel, a, a > zero(a), "a > 0")
        @check_args(PolynomialKernel, c, c >= zero(c), "c ≧ 0")
        @check_args(PolynomialKernel, d, d >= one(d) && d == trunc(d), "d ∈ ℤ₊")
        return new{T,U}(a, c, d)
    end
end
function PolynomialKernel{T}(a::Real,c::Real,d::Real) where {T<:AbstractFloat}
    return PolynomialKernel{T,promote_int()}(a, c, d)
end
function PolynomialKernel(
        a::T₁=1.0,
        c::T₂=T₁(1),
        d::U₁=3
    ) where {T₁<:Real,T₂<:Real,U₁<:Real}
    T = promote_float(T₁,T₂)
    U = promote_int(U₁)
    return PolynomialKernel{T,U}(a, c, d)
end

@inline eltypes(::Type{<:PolynomialKernel{T,U}}) where {T,U} = (T,U)

@inline basefunction(::PolynomialKernel) = ScalarProduct()
@inline function kappa(κ::PolynomialKernel{T}, xᵀy::T) where {T}
    return (κ.a*xᵀy + κ.c)^(κ.d)
end