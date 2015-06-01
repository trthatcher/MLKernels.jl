#==========================================================================
  Periodic Kernel
==========================================================================#

immutable PeriodicKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    period::T
    #ell::Vector{T}
    ell::T
    #function PeriodicKernel(period::T, ell::Vector{T})
    function PeriodicKernel(period::T, ell::T)
        period > 0 || throw(ArgumentError("period = $(period) must be greater than zero."))
        #all(ell .> 0) || throw(ArgumentError("all ell = $(ell) must be greater than zero."))
        ell > 0 || throw(ArgumentError("all ell = $(ell) must be greater than zero."))
        new(period, ell)
    end
end
#PeriodicKernel{T<:FloatingPoint}(period::T, ell::Vector{T}) = PeriodicKernel{T}(period, ell)
#PeriodicKernel{T<:FloatingPoint}(period::T, ell::T, N::Integer) = PeriodicKernel{T}(period, ell*ones(T,N))
#PeriodicKernel{T<:FloatingPoint}(period::T, ell::T) = PeriodicKernel{T}(period, [ell])
PeriodicKernel{T<:FloatingPoint}(period::T, ell::T) = PeriodicKernel{T}(period, ell)

kappa{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = exp(-2sin(π*z/κ.period)^2 / κ.ell^2)
kappa_dz{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = -2sin(2π*z/κ.period) * π/κ.period / κ.ell^2 * kappa(κ, z)
#kappa_dz2{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) =  -2π/κ.period / κ.ell^2 * (cos(2π*z/κ.period)*(2π/κ.period) * kappa(κ, z) + sin(2π*z/κ.period) * kappa_dz(κ, z))
kappa_dz2{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) =  -(2π/(κ.period * κ.ell))^2 * (cos(2π*z/κ.period) - (sin(2π*z/κ.period) / κ.ell)^2) * kappa(κ, z)
kappa_dperiod{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = 2sin(2π*z/κ.period) / κ.ell^2 * π*z/κ.period^2 * kappa(κ, z)
kappa_dell{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = 4sin(π*z/κ.period)^2 / κ.ell^3 * kappa(κ, z)

function kappa_dp{T<:FloatingPoint}(κ::PeriodicKernel{T}, param::Symbol, z::T)
    param == :period ? kappa_dperiod(κ, z) :
    param == :ell    ? kappa_dell(κ, z)    :
                       zero(T)
end

function description_string{T<:FloatingPoint}(κ::PeriodicKernel{T}, eltype::Bool = true)
    "PeriodicKernel" * (eltype ? "{$(T)}" : "") * "(period=$(κ.period),ell=$(κ.ell))"
end

function description_string_long(::PeriodicKernel)
    """
    Periodic Kernel:
    """
end

