#==========================================================================
  Periodic Kernel
==========================================================================#

immutable PeriodicKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    p::T
    ell::T
    function PeriodicKernel(p::T, ell::T)
        p > 0 || throw(ArgumentError("p = $(p) must be greater than zero."))
        ell > 0 || throw(ArgumentError("ell = $(ell) must be greater than zero."))
        new(p, ell)
    end
end
PeriodicKernel{T<:FloatingPoint}(p::T = 1.0, ell::T = 1.0) = PeriodicKernel{T}(p, ell)

kappa{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = exp(-2sin(π*z/κ.p)^2 / κ.ell^2)
kappa_dz{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = -2sin(2π*z/κ.p) * π/κ.p / κ.ell^2 * kappa(κ, z)
#kappa_dz2{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) =  -2π/κ.p / κ.ell^2 * (cos(2π*z/κ.p)*(2π/κ.p) * kappa(κ, z) + sin(2π*z/κ.p) * kappa_dz(κ, z))
kappa_dz2{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) =  -(2π/(κ.p * κ.ell))^2 * (cos(2π*z/κ.p) - (sin(2π*z/κ.p) / κ.ell)^2) * kappa(κ, z)
kappa_dp{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = 2sin(2π*z/κ.p) / κ.ell^2 * π*z/κ.p^2 * kappa(κ, z)
kappa_dell{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = 4sin(π*z/κ.p)^2 / κ.ell^3 * kappa(κ, z)

function kappa_dp{T<:FloatingPoint}(κ::PeriodicKernel{T}, param::Symbol, z::T)
    param == :p   ? kappa_dp(κ, z)   :
    param == :ell ? kappa_dell(κ, z) :
                    zero(T)
end

function description_string{T<:FloatingPoint}(κ::PeriodicKernel{T}, eltype::Bool = true)
    "PeriodicKernel" * (eltype ? "{$(T)}" : "") * "(p=$(κ.p),l=$(κ.ell))"
end

function description_string_long(::PeriodicKernel)
    """
    Periodic Kernel:
    """
end

