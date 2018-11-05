@doc raw"""
    PowerKernel([γ=1])

The Power Kernel is a negative definite kernel given by:

```math
\kappa(\mathbf{x},\mathbf{y}) = 
\|\mathbf{x} - \mathbf{y} \|^{2\gamma}
\qquad \gamma \in (0,1]
```
"""
struct PowerKernel{T<:AbstractFloat} <: NegativeDefiniteKernel{T}
    gamma::HyperParameter{T}
    PowerKernel{T}(γ::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
PowerKernel(γ::T1 = 1.0) where {T1<:Real} = PowerKernel{floattype(T1)}(γ)

@inline powerkernel(z::T, γ::T) where {T<:AbstractFloat} = z^γ

@inline basefunction(::PowerKernel) = SquaredEuclidean()
@inline kappa(κ::PowerKernel{T}, z::T) where {T} = powerkernel(z, getvalue(κ.gamma))