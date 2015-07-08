abstract Kernel{T}
abstract BaseKernel{T<:FloatingPoint,RANGE} <: Kernel{T}

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


abstract AdditiveKernel{T<:FloatingPoint} <: BaseKernel{T}

immutable SquaredDistanceKernel{T<:FloatingPoint} <: AdditiveKernel{T} end
kappa{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::T, y::T) = (x-y)^2
rangemin(::SquaredDistanceKernel) = 0

immutable SineSquaredKernel{T<:FloatingPoint} <: AdditiveKernel{T} end
kappa{T<:FloatingPoint}(κ::SineSquaredKernel{T}, x::T, y::T) = sin(x-y)^2
rangemin(::SineSquaredKernel) = 0

immutable ChiSquaredKernel{T<:FloatingPoint} <: AdditiveKernel{T} end
kappa{T<:FloatingPoint}(κ::ChiSquaredKernel{T}, x::T, y::T) = (x - y)^2/(x + y)
rangemin(::ChiSquaredKernel) = 0


abstract SeparableKernel{T<:FloatingPoint} <: BaseKernel{T}

immutable ScalarProductKernel{T<:FloatingPoint} <: SeparableKernel{T} end
kappa{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::T) = x

immutable MercerSigmoidKernel{T<:FloatingPoint} <: SeparableKernel{T}
    d::T
    b::T
    function MercerSigmoidKernel(d::T, b::T)
        b > 0 || throw(ArgumentError("b = $(b) must be greater than zero."))
        new(d, b)
    end
end
MercerSigmoidKernel{T<:FloatingPoint}(d::T = 0.0, b::T = one(T)) = MercerSigmoidKernel{T}(d, b)
