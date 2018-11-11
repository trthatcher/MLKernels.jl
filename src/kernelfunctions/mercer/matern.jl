@doc raw"""
    MaternKernel([ν=1 [,θ=1]])

The Matern kernel is a Mercer kernel given by:

```math
\kappa(\mathbf{x},\mathbf{y}) =
\frac{1}{2^{\nu-1}\Gamma(\nu)}
\left(\frac{\sqrt{2\nu}||\mathbf{x}-\mathbf{y}||}{\theta}\right)^{\nu}
K_{\nu}\left(\frac{\sqrt{2\nu}||\mathbf{x}-\mathbf{y}||}{\theta}\right)
```

where ``\Gamma`` is the gamma function, ``K_{\nu}`` is the modified Bessel function of the
second kind, ``\nu > 0`` and ``\theta > 0``.
"""
struct MaternKernel{T<:AbstractFloat} <: MercerKernel{T}
    nu::HyperParameter{T}
    rho::HyperParameter{T}
    MaternKernel{T}(ν::Real, ρ::Real) where {T<:AbstractFloat}  = new{T}(
        HyperParameter(convert(T,ν), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,ρ), interval(OpenBound(zero(T)), nothing))
    )
end
function MaternKernel(ν::T1=1.0, ρ::T2=one(T1)) where {T1<:Real,T2<:Real}
    MaternKernel{floattype(T1,T2)}(ν,ρ)
end

@inline function maternkernel(d²::T, ν::T, ρ::T) where {T<:AbstractFloat}
    d = √(d²)
    d = d < eps(T) ? eps(T) : d  # If d is zero, besselk will return NaN
    tmp = √(2ν)d/ρ
    (2^(1 - ν))*(tmp^ν)*besselk(ν, tmp)/gamma(ν)
end

@inline basefunction(::MaternKernel) = SquaredEuclidean()
@inline function kappa(κ::MaternKernel{T}, z::T) where {T}
    maternkernel(z, getvalue(κ.nu), getvalue(κ.rho))
end