@doc raw"""
    MaternKernel([ν=1 [, θ=1]])

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
    ν::T
    ρ::T
    function MaternKernel{T}(ν::Real, ρ::Real) where {T<:AbstractFloat}
        @check_args(MaternKernel, ν, ν > zero(T), "ν > 0")
        @check_args(MaternKernel, ρ, ρ > zero(T), "ρ > 0")
        return new{T}(ν, ρ)
    end
end
function MaternKernel(ν::T₁=1.0, ρ::T₂=one(T1)) where {T₁<:Real,T₂<:Real}
    MaternKernel{floattype(T₁, T₂)}(ν,ρ)
end

@inline basefunction(::MaternKernel) = SquaredEuclidean()

@inline function kappa(κ::MaternKernel{T}, d²::T) where {T}
    d = √(d²)
    d = d < eps(T) ? eps(T) : d  # If d is zero, besselk will return NaN
    tmp = √(2κ.ν)*d/κ.ρ
    return (convert(T, 2)^(one(T) - κ.ν))*(tmp^κ.ν)*besselk(κ.ν, tmp)/gamma(κ.ν)
end