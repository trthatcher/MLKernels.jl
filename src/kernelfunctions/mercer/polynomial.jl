@doc raw"""
    PolynomialKernel([a=1 [,c=1 [,d=3]]])

The polynomial kernel is a Mercer kernel given by:

```math
\kappa(\mathbf{x},\mathbf{y}) = 
(a \mathbf{x}^\intercal \mathbf{y} + c)^d
\qquad \alpha > 0, \; c \geq 0, \; d \in \mathbb{Z}_{+}
```
"""
struct PolynomialKernel{T<:AbstractFloat,U<:Integer} <: MercerKernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    d::HyperParameter{U}
    function PolynomialKernel{T}(a::Real, c::Real, d::U) where {T<:AbstractFloat,U<:Integer}
        new{T,U}(HyperParameter(convert(T,a), interval(OpenBound(zero(T)), nothing)),
                 HyperParameter(convert(T,c), interval(ClosedBound(zero(T)), nothing)),
                 HyperParameter(d, interval(ClosedBound(one(U)), nothing)))
    end
end
function PolynomialKernel(a::T1=1.0, c::T2=one(T1), d::Integer=3) where {T1<:Real,T2<:Real}
    PolynomialKernel{floattype(T1,T2)}(a, c, d)
end

@inline eltypes(::Type{<:PolynomialKernel{T,U}}) where {T,U} = (T,U)
@inline thetafieldnames(κ::PolynomialKernel) = Symbol[:a, :c]

@inline polynomialkernel(z::T, a::T, c::T, d::U) where {T<:AbstractFloat,U<:Integer} = (a*z + c)^d

@inline basefunction(::PolynomialKernel) = ScalarProduct()
@inline function kappa(κ::PolynomialKernel{T}, z::T) where {T}
    polynomialkernel(z, getvalue(κ.a), getvalue(κ.c), getvalue(κ.d))
end