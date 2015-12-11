#==========================================================================
  Squared Distance Kernel
  k(x,y) = (x-y)²ᵗ    x ∈ ℝ, y ∈ ℝ, t ∈ (0,1]
==========================================================================#

doc"`SquaredDistanceKernel(t)` = Σⱼ(xⱼ-yⱼ)²ᵗ"
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

isnegdef(::SquaredDistanceKernel) = true

attainsnegative(::SquaredDistanceKernel) = false

function description_string{T<:AbstractFloat}(κ::SquaredDistanceKernel{T}, eltype::Bool = true)
    "SquaredDistance" * (eltype ? "{$(T)}" : "") * "(t=$(κ.t))"
end

@inline phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T,:t1}, x::T, y::T) = (x-y)^2
@inline phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T,:t0p5}, x::T, y::T) = abs(x-y)
@inline phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T}, x::T, y::T) = ((x-y)^2)^κ.t


#==========================================================================
  Sine Squared Kernel
  k(x,y) = sin²ᵗ(p(x-y))    x ∈ ℝ, y ∈ ℝ, t ∈ (0,1], p ∈ (0,∞)
==========================================================================#

immutable SineSquaredKernel{T<:AbstractFloat,CASE} <: AdditiveKernel{T}
    p::T
    t::T
    function SineSquaredKernel(p::T, t::T)
        0 < p || error("Parameter p = $(p) must be positive.")
        0 < t <= 1 || error("Parameter t = $(t) must be in range (0,1]")
        new(p, t)
    end
end
function SineSquaredKernel{T<:AbstractFloat}(p::T = convert(Float64, π), t::T = one(T))
    CASE =  if t == 1
                :t1
            elseif t == 0.5
                :t0p5
            else
                :∅
            end
    SineSquaredKernel{T,CASE}(p, t)
end

isnegdef(::SineSquaredKernel) = true

attainsnegative(::SineSquaredKernel) = false

function description_string{T<:AbstractFloat}(κ::SineSquaredKernel{T}, eltype::Bool = true)
    "SineSquared" * (eltype ? "{$(T)}" : "") * "(p=$(κ.p),t=$(κ.t))"
end

@inline phi{T<:AbstractFloat}(κ::SineSquaredKernel{T,:t1}, x::T, y::T) = sin(κ.p*(x-y))^2
@inline phi{T<:AbstractFloat}(κ::SineSquaredKernel{T,:t0p5}, x::T, y::T) = abs(sin(κ.p*(x-y)))
@inline phi{T<:AbstractFloat}(κ::SineSquaredKernel{T}, x::T, y::T) = (sin(κ.p*(x-y))^2)^κ.t


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

isnegdef(::ChiSquaredKernel) = true

attainsnegative(::ChiSquaredKernel) = false

function description_string{T<:AbstractFloat}(κ::ChiSquaredKernel{T}, eltype::Bool = true)
    "ChiSquared" * (eltype ? "{$(T)}" : "") * "(t=$(κ.t))"
end

@inline function phi{T<:AbstractFloat}(κ::ChiSquaredKernel{T,:t1}, x::T, y::T)
    (x == y == zero(T)) ? zero(T) : (x-y)^2/(x+y)
end
@inline function phi{T<:AbstractFloat}(κ::ChiSquaredKernel{T},x::T, y::T)
    (x == y == zero(T)) ? zero(T) : ((x-y)^2/(x+y))^κ.t
end


#==========================================================================
  Scalar Product Kernel
==========================================================================#

immutable ScalarProductKernel{T<:AbstractFloat} <: AdditiveKernel{T} end
ScalarProductKernel() = ScalarProductKernel{Float64}()

ismercer(::ScalarProductKernel) = true

function description_string{T<:AbstractFloat}(κ::ScalarProductKernel{T}, eltype::Bool = true)
    "ScalarProduct" * (eltype ? "{$(T)}" : "") * "()"
end

@inline phi{T<:AbstractFloat}(κ::ScalarProductKernel{T}, x::T, y::T) = x*y
