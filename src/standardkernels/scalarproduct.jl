#===================================================================================================
  Scalar Product Kernel Definitions
===================================================================================================#

#==========================================================================
  Polynomial Kernel
  k(x,y) = (αxᵀy + c)ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0, d > 0
==========================================================================#

immutable PolynomialKernel{T<:FloatingPoint,CASE} <: ScalarProductKernel{T}
    alpha::T
    c::T
    d::T
    function PolynomialKernel(α::T, c::T, d::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        c >= 0 || throw(ArgumentError("c = $(c) must be non-negative."))
        (d > 0 && trunc(d) == d) || throw(ArgumentError("d = $(d) must be an integer greater than zero."))
        if CASE == :d1 && d != 1
            error("Special case d = 1 flagged but d = $(convert(Int64,d))")
        end
        new(α, c, d)
    end
end
PolynomialKernel{T<:FloatingPoint}(α::T = 1.0, c::T = one(T), d::T = convert(T, 2)) = PolynomialKernel{T, d == 1 ? :d1 : :Ø}(α, c, d)
PolynomialKernel{T<:FloatingPoint}(α::T, c::T, d::Integer) = PolynomialKernel(α, c, convert(T, d))

LinearKernel{T<:FloatingPoint}(α::T, c::T) = PolynomialKernel(α, c, 1)

ismercer(::PolynomialKernel) = true

function description_string{T<:FloatingPoint}(κ::PolynomialKernel{T}, eltype::Bool = true) 
    "PolynomialKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),c=$(κ.c),d=$(convert(Int64,κ.d)))"
end

kappa{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = (κ.alpha*xᵀy + κ.c)^κ.d
kappa{T<:FloatingPoint}(κ::PolynomialKernel{T,:d1}, xᵀy::T) = κ.alpha*xᵀy + κ.c


#==========================================================================
  Sigmoid Kernel
  k(x,y) = tanh(αxᵀy + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0
==========================================================================#

immutable SigmoidKernel{T<:FloatingPoint} <: ScalarProductKernel{T}
    alpha::T
    c::T
    function SigmoidKernel(α::T, c::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        c >= 0 || throw(ArgumentError("c = $(c) must be non-negative."))
        new(α, c)
    end
end
SigmoidKernel{T<:FloatingPoint}(α::T = 1.0, c::T = one(T)) = SigmoidKernel{T}(α, c)

function description_string{T<:FloatingPoint}(κ::SigmoidKernel{T}, eltype::Bool = true)
    "SigmoidKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),c=$(κ.c))"
end

kappa{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = tanh(κ.alpha*xᵀy + κ.c)
