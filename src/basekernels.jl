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
                :∅
            end
    SquaredDistanceKernel{Float64,CASE}(t)
end

rangemin(::SquaredDistanceKernel) = 0
isnegdef(::SquaredDistanceKernel) = true

function description_string{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, eltype::Bool = true)
    "SquaredDistanceKernel" * (eltype ? "{$(T)}" : "") * "(t=$(κ.t))"
end

kappa{T<:FloatingPoint}(κ::SquaredDistanceKernel{T,:t1}, x::T, y::T) = (x-y)^2
kappa{T<:FloatingPoint}(κ::SquaredDistanceKernel{T,:t0p5}, x::T, y::T) = abs(x-y)
kappa{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::T, y::T) = ((x-y)^2)^κ.t


#==========================================================================
  Sine Squared Kernel
  k(x,y) = sin²ᵗ(x-y)    x ∈ ℝ, y ∈ ℝ, t ∈ (0,1]
==========================================================================#

immutable SineSquaredKernel{T<:FloatingPoint,CASE} <: AdditiveKernel{T}
    t::T
    function SineSquaredKernel(t::T)
        0 < t <= 1 || error("Bad range")
        new(t)
    end
end
function SineSquaredKernel{T<:FloatingPoint}(t::T = 1.0)
    CASE =  if t == 1
                :t1
            elseif t == 0.5
                :t0p5
            else
                :∅
            end
    SineSquaredKernel{Float64,CASE}(t)
end

rangemin(::SineSquaredKernel) = 0
isnegdef(::SineSquaredKernel) = true

function description_string{T<:FloatingPoint}(κ::SineSquaredKernel{T}, eltype::Bool = true)
    "SineSquaredKernel" * (eltype ? "{$(T)}" : "") * "(t=$(κ.t))"
end

kappa{T<:FloatingPoint}(κ::SineSquaredKernel{T,:t1}, x::T, y::T) = sin(x-y)^2
kappa{T<:FloatingPoint}(κ::SineSquaredKernel{T,:t0p5}, x::T, y::T) = abs(sin(x-y))
kappa{T<:FloatingPoint}(κ::SineSquaredKernel{T}, x::T, y::T) = (sin(x-y)^2)^κ.t


#==========================================================================
  Chi Squared Kernel
  k(x,y) = (x-y)²ᵗ/(x+y)    x ∈ ℝ⁺, y ∈ ℝ⁺, t ∈ (0,1]
==========================================================================#

immutable ChiSquaredKernel{T<:FloatingPoint,CASE} <: AdditiveKernel{T}
    t::T
    function ChiSquaredKernel(t::T)
        0 < t <= 1 || error("Bad range")
        new(t)
    end
end
function ChiSquaredKernel{T<:FloatingPoint}(t::T = 1.0)
    CASE =  if t == 1
                :t1
            else
                :∅
            end
    ChiSquaredKernel{Float64,CASE}(t)
end

rangemin(::ChiSquaredKernel) = 0
isnegdef(::ChiSquaredKernel) = true

function description_string{T<:FloatingPoint}(κ::ChiSquaredKernel{T}, eltype::Bool = true)
    "ChiSquaredKernel" * (eltype ? "{$(T)}" : "") * "(t=$(κ.t))"
end

kappa{T<:FloatingPoint}(κ::ChiSquaredKernel{T,:t1}, x::T, y::T) = (x-y)^2/(x+y)
kappa{T<:FloatingPoint}(κ::ChiSquaredKernel{T}, x::T, y::T) = ((x-y)^2/(x+y))^κ.t


#==========================================================================
  Separable Kernel
  k(x,y) = k(x)k(y)    x ∈ ℝ, y ∈ ℝ
==========================================================================#

abstract SeparableKernel{T<:FloatingPoint} <: AdditiveKernel{T}

kappa{T<:FloatingPoint}(κ::SeparableKernel{T}, x::T, y::T) = kappa(κ,x) * kappa(κ,y)

#==========================================================================
  Scalar Product Kernel
==========================================================================#

immutable ScalarProductKernel{T<:FloatingPoint} <: SeparableKernel{T} end
ScalarProductKernel() = ScalarProductKernel{Float64}()

ismercer(::ScalarProductKernel) = true

function description_string{T<:FloatingPoint}(κ::ScalarProductKernel{T}, eltype::Bool = true)
    "ScalarProduct" * (eltype ? "{$(T)}" : "") * "()"
end

kappa{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::T) = x


#==========================================================================
  Mercer Sigmoid Kernel
==========================================================================#

immutable MercerSigmoidKernel{T<:FloatingPoint} <: SeparableKernel{T}
    d::T
    b::T
    function MercerSigmoidKernel(d::T, b::T)
        b > 0 || throw(ArgumentError("b = $(b) must be greater than zero."))
        new(d, b)
    end
end
MercerSigmoidKernel{T<:FloatingPoint}(d::T = 0.0, b::T = one(T)) = MercerSigmoidKernel{T}(d, b)

ismercer(::MercerSigmoidKernel) = true

function description_string{T<:FloatingPoint}(κ::MercerSigmoidKernel{T}, eltype::Bool = true)
    "MercerSigmoidProduct" * (eltype ? "{$(T)}" : "") * "(d=$(κ.d),b=$(κ.b))"
end

kappa{T<:FloatingPoint}(κ::MercerSigmoidKernel{T}, x::T) = tanh((x-d)/b)
