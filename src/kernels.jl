abstract Kernel{T}

ismercer(::Kernel) = false
isnegdef(::Kernel) = false

rangemax(::Kernel) = Inf
rangemin(::Kernel) = -Inf
attainsrangemax(::Kernel) = true
attainsrangemin(::Kernel) = true

<=(κ::Kernel, x::Real) = attainsrangemax(κ) ? (rangemax(κ) <= x) : (rangemax(κ) <= x)
<=(x::Real, κ::Kernel) = attainsrangemin(κ) ? (x <= rangemin(κ)) : (x <  rangemin(κ))

<(κ::Kernel, x::Real)  = attainsrangemax(κ) ? (rangemax(κ) <= x) : (rangemax(κ) <  x)
<(x::Real, κ::Kernel)  = attainsrangemin(κ) ? (x <  rangemin(κ)) : (x <= rangemax(κ))

>=(κ::Kernel, x::Real) = x <= κ
>=(x::Real, κ::Kernel) = κ <= x

>(κ::Kernel, x::Real)  = x < κ
>(x::Real, κ::Kernel)  = κ < x

#==========================================================================
  Base Kernels
==========================================================================#

include("basekernels.jl")


#==========================================================================
  ARD Kernel
==========================================================================#

immutable ARD{T<:FloatingPoint} <: BaseKernel{T}
    k::AdditiveKernel{T}
    w::Vector{T}
    function ARD(κ::AdditiveKernel{T}, w::Vector{T})
        all(w .> 0) || error("Weights must be positive real numbers.")
        new(κ, w)
    end
end
ARD{T<:FloatingPoint}(κ::AdditiveKernel{T}, w::Vector{T}) = ARD{T}(κ, w)

ismercer(κ::ARD) = ismercer(κ.k)
isnegdef(κ::ARD) = isnegdef(κ.k)
