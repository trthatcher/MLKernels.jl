@doc raw"""
    LogKernel([α [,γ]])

The Log Kernel is a negative definite kernel given by:

```math
\kappa(\mathbf{x},\mathbf{y}) = 
\log \left(1 + \alpha\|\mathbf{x} - \mathbf{y} \|^{2\gamma}\right)
\qquad \alpha > 0, \; \gamma \in (0,1]
```
"""
struct LogKernel{T<:AbstractFloat} <: NegativeDefiniteKernel{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    LogKernel{T}(α::Real, γ::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
function LogKernel(α::T1 = 1.0, γ::T2 = one(T1)) where {T1<:Real,T2<:Real}
    LogKernel{floattype(T1,T2)}(α, γ)
end

@inline logkernel(z::T, α::T, γ::T) where {T<:AbstractFloat} = log(α*z^γ+1)

@inline basefunction(::LogKernel) = SquaredEuclidean()
@inline function kappa(κ::LogKernel{T}, z::T) where {T}
    logkernel(z, getvalue(κ.alpha), getvalue(κ.gamma))
end