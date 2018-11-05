@doc raw"""
    LinearKernel([a=1 [,c=1]])

The linear kernel is a Mercer kernel given by:

```math
\kappa(\mathbf{x},\mathbf{y}) = 
a \mathbf{x}^\intercal \mathbf{y} + c \qquad \alpha > 0, \; c \geq 0
```
"""
struct LinearKernel{T<:AbstractFloat} <: MercerKernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    LinearKernel{T}(a::Real, c::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,a), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,c), interval(ClosedBound(zero(T)), nothing))
    )
end
LinearKernel(a::T1=1.0, c::T2=one(T1)) where {T1<:Real,T2<:Real} = LinearKernel{floattype(T1,T2)}(a,c)

@inline linearkernel(z::T, a::T, c::T) where {T<:AbstractFloat} = a*z + c

@inline basefunction(::LinearKernel) = ScalarProduct()
@inline kappa(κ::LinearKernel{T}, z::T) where {T} = linearkernel(z, getvalue(κ.a), getvalue(κ.c))
