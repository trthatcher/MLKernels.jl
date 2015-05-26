#===================================================================================================
  Scalar Product Kernel Definitions
===================================================================================================#

#== Linear Kernel ====================#

immutable LinearKernel{T<:FloatingPoint} <: ScalarProductKernel{T}
    c::T
    function LinearKernel(c::T)
        c >= 0 || throw(ArgumentError("c = $c must be greater than zero."))
        new(c)
    end
end
LinearKernel{T<:FloatingPoint}(c::T = 1.0) = LinearKernel{T}(c)

kappa{T<:FloatingPoint}(κ::LinearKernel{T}, xᵀy::T) = xᵀy + κ.c
kappa_dz{T<:FloatingPoint}(κ::LinearKernel{T}, xᵀy::T) = one(T)
kappa_dz2{T<:FloatingPoint}(κ::LinearKernel{T}, xᵀy::T) = zero(T)
kappa_dc{T<:FloatingPoint}(κ::LinearKernel{T}, xᵀy::T) = one(T)

function kappa_dp{T<:FloatingPoint}(κ::LinearKernel{T}, param::Symbol, z::T)
    param == :c ? kappa_dc(κ, z) : zero(T)
end

isposdef(::LinearKernel) = true

function description_string{T<:FloatingPoint}(κ::LinearKernel{T}, eltype::Bool = true)
    "LinearKernel" * (eltype ? "{$(T)}" : "") * "(c=$(κ.c))"
end

function description_string_long(::LinearKernel)
    """ 
    Linear Kernel:

    The linear kernel differs from the ordinary inner product by the
    addition of an optional constant c ≥ 0:

        k(x,y) = xᵀy + c    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0

    Techniques using the linear kernel often do not differ from their
    non-kappad versions.
    """
end


#== Polynomial Kernel ===============#

immutable PolynomialKernel{T<:FloatingPoint} <: ScalarProductKernel{T}
    alpha::T
    c::T
    d::T
    function PolynomialKernel(α::T, c::T, d::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        c >= 0 || throw(ArgumentError("c = $(c) must be a non-negative number."))
        d > 0 || throw(ArgumentError("d = $(d) must be greater than zero."))
        new(α, c, d)
    end
end
function PolynomialKernel{T<:FloatingPoint}(α::T = 1.0, c::T = one(T), d::T = convert(T, 2))
    PolynomialKernel{T}(α, c, d)
end

PolynomialKernel{T<:FloatingPoint}(α::T, c::T, d::Integer) = PolynomialKernel(α, c, convert(T, d))

kappa{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = (κ.alpha*xᵀy + κ.c)^κ.d
kappa_dz{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = κ.alpha*κ.d*(κ.alpha*xᵀy + κ.c)^(κ.d-1)
kappa_dz2{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = κ.alpha^2*κ.d*(κ.d-1)*(κ.alpha*xᵀy + κ.c)^(κ.d-2)
kappa_dalpha{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = xᵀy*κ.d*(κ.alpha*xᵀy + κ.c)^(κ.d-1)
kappa_dc{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = κ.d*(κ.alpha*xᵀy + κ.c)^(κ.d-1)
kappa_dd{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = log(κ.alpha*xᵀy + κ.c)*(κ.alpha*xᵀy + κ.c)^κ.d

function kappa_dp{T<:FloatingPoint}(κ::PolynomialKernel{T}, param::Symbol, z::T)
    param == :alpha ? kappa_dalpha(κ, z) :
    param == :c     ? kappa_dc(κ, z) :
    param == :d     ? kappa_dd(κ, z) :
                      zero(T)
end


isposdef(::PolynomialKernel) = true

function description_string{T<:FloatingPoint}(κ::PolynomialKernel{T}, eltype::Bool = true) 
    "PolynomialKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),c=$(κ.c),d=$(κ.d))"
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

kappa{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = tanh(κ.alpha*xᵀy + κ.c)
kappa_dz{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = (1 - kappa(κ,xᵀy)^2) * κ.alpha
kappa_dz2{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = -2κ.alpha^2 * kappa(κ,xᵀy)*(1-kappa(κ,xᵀy)^2)
kappa_dalpha{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = (1 - kappa(κ,xᵀy)^2) * xᵀy
kappa_dc{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = (1 - kappa(κ,xᵀy)^2)

function kappa_dp{T<:FloatingPoint}(κ::SigmoidKernel{T}, param::Symbol, z::T)
    param == :alpha ? kappa_dalpha(κ, z) :
    param == :c     ? kappa_dc(κ, z) :
                      zero(T)
end

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

