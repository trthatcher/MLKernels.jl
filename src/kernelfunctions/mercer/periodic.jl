@doc raw"""
    PeriodicKernel([α=1 [,p=π]])

The periodic kernel is given by:

```math
\kappa(\mathbf{x},\mathbf{y}) =
\exp\left(-\alpha \sum_{i=1}^n \sin(p(x_i - y_i))^2\right)
\qquad p >0, \; \alpha > 0
```

where ``\mathbf{x}`` and ``\mathbf{y}`` are ``n`` dimensional vectors. The parameters ``p`` 
and ``\alpha`` are scaling parameters for the periodicity and the magnitude, respectively. 
This kernel is useful when data has periodicity to it.
"""
struct PeriodicKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    PeriodicKernel{T}(α::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
PeriodicKernel(α::T1 = 1.0) where {T1<:Real} = PeriodicKernel{floattype(T1)}(α)

@inline basefunction(::PeriodicKernel) = SineSquared()
@inline kappa(κ::PeriodicKernel{T}, z::T) where {T} = squaredexponentialkernel(z, getvalue(κ.alpha))