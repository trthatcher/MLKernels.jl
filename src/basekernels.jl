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


#==========================================================================
  Additive Kernel
  k(x,y) = sum(k(x_i,y_i))    x ∈ ℝⁿ, y ∈ ℝⁿ
==========================================================================#

abstract AdditiveKernel{T<:FloatingPoint} <: BaseKernel{T}


#==========================================================================
  Squared Distance Kernel
  k(x,y) = (x-y)²ᵗ    x ∈ ℝ, y ∈ ℝ, t ∈ (0,1]
==========================================================================#

immutable SquaredDistanceKernel{T<:FloatingPoint,CASE} <: AdditiveKernel{T} 
    t::T
    function SquaredDistanceKernel(t::T)
        0 < t <= 1 || error("Bad range")
        new(t)
    end
end
function SquaredDistanceKernel{T<:FloatingPoint}(t::T = 1.0)
    CASE =  if t == 1
                :t1
            elseif t == 0.5
                :t0p5
            else
                :ϕ
            end
    SquaredDistanceKernel{Float64,CASE}(t)
end

rangemin(::SquaredDistanceKernel) = 0
isnegdef(::SquaredDistanceKernel) = true

function description_string{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, eltype::Bool = true)
    "SquaredDistanceKernel" * (eltype ? "{$(T)}" : "") * "(t=$(κ.t)))"
end

kappa{T<:FloatingPoint}(κ::SquaredDistanceKernel{T,:t1}, x::T, y::T) = (x-y)^2
kappa{T<:FloatingPoint}(κ::SquaredDistanceKernel{T,:t0p5}, x::T, y::T) = abs(x-y)
kappa{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::T, y::T) = ((x-y)^2)^κ.t


#==========================================================================
  Sine Squared Kernel
  k(x,y) = sin²ᵗ(x-y)    x ∈ ℝ, y ∈ ℝ, t ∈ (0,1]
==========================================================================#

immutable SineSquaredKernel{T<:FloatingPoint} <: AdditiveKernel{T}
    t::T
    function SineSquaredKernel(t::T)
        0 < t <= 1 || error("Bad range")
        new(t)
    end
end
SineSquaredKernel{T<:FloatingPoint}(t::T = 1.0) = SineSquaredKernel{Float64, t == 1 ? :t1 : :ϕ}()



kappa{T<:FloatingPoint}(κ::SineSquaredKernel{T}, x::T, y::T) = sin(x-y)^2
rangemin(::SineSquaredKernel) = 0


#==========================================================================
  Chi Squared Kernel
  k(x,y) = (x-y)²ᵗ/(x+y)    x ∈ ℝ⁺, y ∈ ℝ⁺, t ∈ (0,1]
==========================================================================#

immutable ChiSquaredKernel{T<:FloatingPoint} <: AdditiveKernel{T} end
kappa{T<:FloatingPoint}(κ::ChiSquaredKernel{T}, x::T, y::T) = (x - y)^2/(x + y)
rangemin(::ChiSquaredKernel) = 0


#==========================================================================
  Separable Kernel
  k(x,y) = (x-y)²ᵗ/(x+y)    x ∈ ℝ⁺, y ∈ ℝ⁺, t ∈ (0,1]
==========================================================================#

abstract SeparableKernel{T<:FloatingPoint,ALGORTHM} <: AdditiveKernel{T}

kappa{T<:FloatingPoint}(κ::SeparableKernel{T}, x::T, y::T) = kappa(κ,x) * kappa(κ,y)



immutable ScalarProductKernel{T<:FloatingPoint,ALGORITHM} <: SeparableKernel{T,ALGORITHM} end
ScalarProductKernel() = ScalarProductKernel{Float64,:Fast}()

kappa{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::T) = x

immutable MercerSigmoidKernel{T<:FloatingPoint,ALGORITHM} <: SeparableKernel{T,ALGORITHM}
    d::T
    b::T
    function MercerSigmoidKernel(d::T, b::T)
        b > 0 || throw(ArgumentError("b = $(b) must be greater than zero."))
        new(d, b)
    end
end
MercerSigmoidKernel{T<:FloatingPoint}(d::T = 0.0, b::T = one(T)) = MercerSigmoidKernel{T,:Fast}(d, b)


#ARD - Weighted additive kernels
immutable ARD{T<:FloatingPoint} <: BaseKernel{T}
    k::AdditiveKernel{T}
    w::Vector{T}
    function ARD(κ::AdditiveKernel{T}, w::Vector{T})
        all(w .> 0) || error("Weights must be positive real numbers.")
        new(κ, w)
    end
end
ARD{T<:FloatingPoint}(κ::Kernel{T}, w::Vector{T}) = ARD{T}(κ, w)
