#==========================================================================
  Additive Kernel
  k(x,y) = sum(k(x_i,y_i))    x ∈ ℝⁿ, y ∈ ℝⁿ
==========================================================================#

abstract AdditiveKernel{T<:AbstractFloat} <: BaseKernel{T}


#==========================================================================
  Squared Distance Kernel
  k(x,y) = (x-y)²ᵗ    x ∈ ℝ, y ∈ ℝ, t ∈ (0,1]
==========================================================================#

immutable SquaredDistanceKernel{T<:AbstractFloat,CASE} <: AdditiveKernel{T} 
    t::T
    function SquaredDistanceKernel(t::T)
        0 < t <= 1 || error("Parameter t = $(t) must be in range (0,1]")
        new(t)
    end
end
function SquaredDistanceKernel{T<:AbstractFloat}(t::T = 1.0)
    CASE =  if t == 1
                :t1
            elseif t == 0.5
                :t0p5
            else
                :∅
            end
    SquaredDistanceKernel{T,CASE}(t)
end

rangemin(::SquaredDistanceKernel) = 0
isnegdef(::SquaredDistanceKernel) = true

function description_string{T<:AbstractFloat}(κ::SquaredDistanceKernel{T}, eltype::Bool = true)
    "SquaredDistance" * (eltype ? "{$(T)}" : "") * "(t=$(κ.t))"
end

convert{T<:AbstractFloat}(::Type{SquaredDistanceKernel{T}}, κ::SquaredDistanceKernel) = SquaredDistanceKernel(convert(T, κ.t))

phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T,:t1}, x::T, y::T) = (x-y)^2
phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T,:t0p5}, x::T, y::T) = abs(x-y)
phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T}, x::T, y::T) = ((x-y)^2)^κ.t


#==========================================================================
  Sine Squared Kernel
  k(x,y) = sin²ᵗ(x-y)    x ∈ ℝ, y ∈ ℝ, t ∈ (0,1]
==========================================================================#

immutable SineSquaredKernel{T<:AbstractFloat,CASE} <: AdditiveKernel{T}
    t::T
    function SineSquaredKernel(t::T)
        0 < t <= 1 || error("Parameter t = $(t) must be in range (0,1]")
        new(t)
    end
end
function SineSquaredKernel{T<:AbstractFloat}(t::T = 1.0)
    CASE =  if t == 1
                :t1
            elseif t == 0.5
                :t0p5
            else
                :∅
            end
    SineSquaredKernel{T,CASE}(t)
end

rangemin(::SineSquaredKernel) = 0
isnegdef(::SineSquaredKernel) = true

function description_string{T<:AbstractFloat}(κ::SineSquaredKernel{T}, eltype::Bool = true)
    "SineSquared" * (eltype ? "{$(T)}" : "") * "(t=$(κ.t))"
end

convert{T<:AbstractFloat}(::Type{SineSquaredKernel{T}}, κ::SineSquaredKernel) = SineSquaredKernel(convert(T, κ.t))

phi{T<:AbstractFloat}(κ::SineSquaredKernel{T,:t1}, x::T, y::T) = sin(x-y)^2
phi{T<:AbstractFloat}(κ::SineSquaredKernel{T,:t0p5}, x::T, y::T) = abs(sin(x-y))
phi{T<:AbstractFloat}(κ::SineSquaredKernel{T}, x::T, y::T) = (sin(x-y)^2)^κ.t


#==========================================================================
  Chi Squared Kernel
  k(x,y) = ((x-y)²/(x+y))ᵗ    x ∈ ℝ⁺, y ∈ ℝ⁺, t ∈ (0,1]
==========================================================================#

immutable ChiSquaredKernel{T<:AbstractFloat,CASE} <: AdditiveKernel{T}
    t::T
    function ChiSquaredKernel(t::T)
        0 < t <= 1 || error("Parameter t = $(t) must be in range (0,1]")
        new(t)
    end
end
function ChiSquaredKernel{T<:AbstractFloat}(t::T = 1.0)
    CASE =  if t == 1
                :t1
            else
                :∅
            end
    ChiSquaredKernel{T,CASE}(t)
end

rangemin(::ChiSquaredKernel) = 0
isnegdef(::ChiSquaredKernel) = true

function description_string{T<:AbstractFloat}(κ::ChiSquaredKernel{T}, eltype::Bool = true)
    "ChiSquared" * (eltype ? "{$(T)}" : "") * "(t=$(κ.t))"
end

convert{T<:AbstractFloat}(::Type{ChiSquaredKernel{T}}, κ::ChiSquaredKernel) = ChiSquaredKernel(convert(T, κ.t))

phi{T<:AbstractFloat}(κ::ChiSquaredKernel{T,:t1}, x::T, y::T) = (x-y)^2/(x+y)
phi{T<:AbstractFloat}(κ::ChiSquaredKernel{T}, x::T, y::T) = ((x-y)^2/(x+y))^κ.t


#==========================================================================
  Separable Kernel
  k(x,y) = k(x)k(y)    x ∈ ℝ, y ∈ ℝ
==========================================================================#

abstract SeparableKernel{T<:AbstractFloat} <: AdditiveKernel{T}

phi{T<:AbstractFloat}(κ::SeparableKernel{T}, x::T, y::T) = phi(κ,x) * phi(κ,y)

#==========================================================================
  Scalar Product Kernel
==========================================================================#

immutable ScalarProductKernel{T<:AbstractFloat} <: SeparableKernel{T} end
ScalarProductKernel() = ScalarProductKernel{Float64}()

ismercer(::ScalarProductKernel) = true

function description_string{T<:AbstractFloat}(κ::ScalarProductKernel{T}, eltype::Bool = true)
    "ScalarProduct" * (eltype ? "{$(T)}" : "") * "()"
end

convert{T<:AbstractFloat}(::Type{ScalarProductKernel{T}}, κ::ScalarProductKernel) = ScalarProductKernel{T}()

phi{T<:AbstractFloat}(κ::ScalarProductKernel{T}, x::T) = x


#==========================================================================
  Mercer Sigmoid Kernel
==========================================================================#

immutable MercerSigmoidKernel{T<:AbstractFloat} <: SeparableKernel{T}
    d::T
    b::T
    function MercerSigmoidKernel(d::T, b::T)
        b > 0 || error("b = $(b) must be greater than zero.")
        new(d, b)
    end
end
MercerSigmoidKernel{T<:AbstractFloat}(d::T = 0.0, b::T = one(T)) = MercerSigmoidKernel{T}(d, b)

ismercer(::MercerSigmoidKernel) = true

function description_string{T<:AbstractFloat}(κ::MercerSigmoidKernel{T}, eltype::Bool = true)
    "MercerSigmoid" * (eltype ? "{$(T)}" : "") * "(d=$(κ.d),b=$(κ.b))"
end

convert{T<:AbstractFloat}(::Type{MercerSigmoidKernel{T}}, κ::MercerSigmoidKernel) = MercerSigmoidKernel{T}(convert(T,κ.d), convert(T,κ.b))

phi{T<:AbstractFloat}(κ::MercerSigmoidKernel{T}, x::T) = tanh((x-κ.d)/κ.b)
