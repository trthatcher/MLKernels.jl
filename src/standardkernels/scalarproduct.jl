#===================================================================================================
  Scalar Product Kernel Definitions
===================================================================================================#

#== Polynomial Kernel ===============#

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

function description_string_long(::PolynomialKernel)
    """ 
    Polynomial Kernel:
     
    The polynomial kernel is a non-stationary kernel which represents
    the original features as in a feature space over polynomials up to 
    degree d of the original variables:

        k(x,y) = (αxᵀy + c)ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0, d > 0

    This kernel is sensitive to numerical instability in the case that
    d is increasingly large and αxᵀy + c approaches zero.
    """
end

kappa{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = (κ.alpha*xᵀy + κ.c)^κ.d
kappa{T<:FloatingPoint}(κ::PolynomialKernel{T,:d1}, xᵀy::T) = κ.alpha*xᵀy + κ.c

kappa_dz{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = κ.alpha*κ.d*(κ.alpha*xᵀy + κ.c)^(κ.d-1)
kappa_dz{T<:FloatingPoint}(κ::PolynomialKernel{T,:d1}, xᵀy::T) = κ.alpha

kappa_dz2{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = κ.alpha^2*κ.d*(κ.d-1)*(κ.alpha*xᵀy + κ.c)^(κ.d-2)

kappa_dalpha{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = xᵀy*κ.d*(κ.alpha*xᵀy + κ.c)^(κ.d-1)

kappa_dc{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = κ.d*(κ.alpha*xᵀy + κ.c)^(κ.d-1)

function kappa_dp{T<:FloatingPoint}(κ::PolynomialKernel{T}, param::Symbol, z::T)
    if param == :alpha
        kappa_dalpha(κ, z)
    elseif param == :c
        kappa_dc(κ, z)
    elseif param == :d
        error("kappa_dd: Cannot differentiate with respect to an integer parameter.")
    else
        zero(T)
    end
end


#== Sigmoid Kernel ===============#

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

ismercer(::SigmoidKernel) = false

function description_string{T<:FloatingPoint}(κ::SigmoidKernel{T}, eltype::Bool = true)
    "SigmoidKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),c=$(κ.c))"
end

function description_string_long(::SigmoidKernel)
    """ 
    Sigmoid Kernel:
     
    The sigmoid kernel is only positive semidefinite. It is used in the
    field of neural networks where it is often used as the activation
    function for artificial neurons.

        k(x,y) = tanh(αxᵀy + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0
    """
end

kappa{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = tanh(κ.alpha*xᵀy + κ.c)

kappa_dz{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = (1 - kappa(κ,xᵀy)^2) * κ.alpha

kappa_dz2{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = -2κ.alpha^2 * kappa(κ,xᵀy)*(1-kappa(κ,xᵀy)^2)

kappa_dalpha{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = (1 - kappa(κ,xᵀy)^2) * xᵀy

kappa_dc{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = (1 - kappa(κ,xᵀy)^2)

function kappa_dp{T<:FloatingPoint}(κ::SigmoidKernel{T}, param::Symbol, z::T)
    if param == :alpha
        kappa_dalpha(κ, z)
    elseif param == :c
        kappa_dc(κ, z)
    else
        zero(T)
    end
end
