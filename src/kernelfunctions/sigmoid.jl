@doc raw"""
    SigmoidKernel([a=1 [,c=1]])
    
The Sigmoid Kernel is given by

```math
\kappa(\mathbf{x},\mathbf{y}) = 
\tanh(a \mathbf{x}^\intercal \mathbf{y} + c) 
\qquad \alpha > 0, \; c \geq 0
```
The sigmoid kernel is a not a true kernel, although it has been used in application. 
"""
struct SigmoidKernel{T<:AbstractFloat} <: Kernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    SigmoidKernel{T}(a::Real, c::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,a), interval(OpenBound(zero(T)),   nothing)),
        HyperParameter(convert(T,c), interval(ClosedBound(zero(T)), nothing))
    )
end
function SigmoidKernel(a::T1 = 1.0, c::T2 = one(T1)) where {T1<:Real,T2<:Real}
    SigmoidKernel{floattype(T1,T2)}(a,c)
end

@inline sigmoidkernel(z::T, a::T, c::T) where {T<:AbstractFloat} = tanh(a*z + c)

@inline basefunction(::SigmoidKernel) = ScalarProduct()
@inline function kappa(κ::SigmoidKernel{T}, z::T) where {T}
    sigmoidkernel(z, getvalue(κ.a), getvalue(κ.c))
end