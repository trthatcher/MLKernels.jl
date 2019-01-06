# Removal of LinearKernel
Base.@deprecate LinearKernel(a::Real=1, c::Real=1) PolynomialKernel(a, c, 1)

# Renaming of GammaRational to GammaRationalQuadratic
Base.@deprecate GammaRationalKernel GammaRationalQuadraticKernel

# Removal of PeriodicKernel
struct SineSquared <: PreMetric end
@inline base_aggregate(::SineSquared, s::T, x::T, y::T) where {T} = s + sin(x-y)^2
@inline isstationary(::SineSquared) = true

struct PeriodicKernel{T<:AbstractFloat} <: MercerKernel{T}
    α::T
    function PeriodicKernel{T}(α::Real) where {T<:AbstractFloat}
        Base.depwarn("PeriodicKernel will be removed in the next major release", :PeriodicKernel)
        @check_args(PeriodicKernel, α, α > zero(α), "α > 0")
        new{T}(α)
    end
end
PeriodicKernel(α::T₁ = 1.0) where {T₁<:Real} = PeriodicKernel{promote_float(T₁)}(α)

@inline basefunction(::PeriodicKernel) = SineSquared()

@inline function kappa(κ::PeriodicKernel{T}, z::T) where {T}
    return exp(-κ.α*z)
end