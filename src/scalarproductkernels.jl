#===================================================================================================
  Scalar Product Kernels
===================================================================================================#

abstract ScalarProductKernel{T<:FloatingPoint} <: StandardKernel{T}

# xᵀy
function scalar_product{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    n = length(x)
    BLAS.dot(n, x, 1, y, 1)
end

# k(x,y) = f(xᵀy)
function kernel_function{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::Array{T}, y::Array{T})
    kernelize_scalar(κ, scalar_product(x, y))
end


#== Linear Kernel ====================#

immutable LinearKernel{T<:FloatingPoint} <: ScalarProductKernel{T}
    c::T
    function LinearKernel(c::T)
        c >= 0 || throw(ArgumentError("c = $c must be greater than zero."))
        new(c)
    end
end
LinearKernel{T<:FloatingPoint}(c::T = 1.0) = LinearKernel{T}(c)

function convert{T<:FloatingPoint}(::Type{LinearKernel{T}}, κ::LinearKernel)
    LinearKernel(convert(T, κ.c))
end

kernelize_scalar{T<:FloatingPoint}(κ::LinearKernel{T}, xᵀy::T) = xᵀy + κ.c

isposdef_kernel(κ::LinearKernel) = true

function description_string{T<:FloatingPoint}(κ::LinearKernel{T}, eltype::Bool = true)
    "LinearKernel" * (eltype ? "{$(T)}" : "") * "(c=$(κ.c))"
end

function description(κ::LinearKernel)
    print(
        """ 
         Linear Kernel:
         
         The linear kernel differs from the ordinary inner product by the
         addition of an optional constant c ≥ 0:

             k(x,y) = xᵀy + c    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0

         Techniques using the linear kernel often do not differ from their
         non-kernelized versions.
        """
    )
end


#== Polynomial Kernel ===============#

immutable PolynomialKernel{T<:FloatingPoint} <: ScalarProductKernel{T}
    α::T
    c::T
    d::T
    function PolynomialKernel(α::T, c::T, d::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        c >= 0 || throw(ArgumentError("c = $(c) must be a non-negative number."))
        d > 0 || throw(ArgumentError("d = $(d) must be a positive integer."))
        b = trunc(d)
        d == b || warn("d = $(d) was truncated to $(b).")
        new(α, c, b)
    end
end
function PolynomialKernel{T<:FloatingPoint}(α::T = 1.0, c::T = one(T), d::T = convert(T, 2))
    PolynomialKernel{T}(α, c, d)
end

PolynomialKernel{T<:FloatingPoint}(α::T, c::T, d::Integer) = PolynomialKernel(α, c, convert(T, d))

function convert{T<:FloatingPoint}(::Type{PolynomialKernel{T}}, κ::PolynomialKernel)
    PolynomialKernel(convert(T, κ.α), convert(T, κ.c), convert(T, κ.d))
end

function kernelize_scalar{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T)
    (κ.α*xᵀy + κ.c)^κ.d
end

isposdef_kernel(κ::PolynomialKernel) = true

function description_string{T<:FloatingPoint}(κ::PolynomialKernel{T}, eltype::Bool = true) 
    "PolynomialKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.α),c=$(κ.c),d=$(κ.d))"
end

function description(κ::PolynomialKernel)
    print(
        """ 
         Polynomial Kernel:
         
         The polynomial kernel is a non-stationary kernel which represents
         the original features as in a feature space over polynomials up to 
         degree d of the original variables:

             k(x,y) = (αxᵀy + c)ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0, d > 0

         This kernel is sensitive to numerical instability in the case that
         d is increasingly large and αxᵀy + c approaches zero.
        """
    )
end


#== Sigmoid Kernel ===============#

immutable SigmoidKernel{T<:FloatingPoint} <: ScalarProductKernel{T}
    α::T
    c::T
    function SigmoidKernel(α::T, c::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        c >= 0 || throw(ArgumentError("c = $(c) must be non-negative."))
        new(α, c)
    end
end
SigmoidKernel{T<:FloatingPoint}(α::T = 1.0, c::T = one(T)) = SigmoidKernel{T}(α, c)

function convert{T<:FloatingPoint}(::Type{SigmoidKernel{T}}, κ::SigmoidKernel)
    SigmoidKernel(convert(T, κ.α), convert(T, κ.c))
end

kernelize_scalar{T<:FloatingPoint}(κ::SigmoidKernel{T}, xᵀy::T) = tanh(κ.α*xᵀy + κ.c)

function description_string{T<:FloatingPoint}(κ::SigmoidKernel{T}, eltype::Bool = true)
    "SigmoidKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.α),c=$(κ.c))"
end

function description(κ::SigmoidKernel)
    print(
        """ 
         Sigmoid Kernel:
         
         The sigmoid kernel is only positive semidefinite. It is used in the
         field of neural networks where it is often used as the activation
         function for artificial neurons.

             k(x,y) = tanh(αxᵀy + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0
        """
    )
end


#==========================================================================
  Conversions
==========================================================================#

for kernel in (:LinearKernel, :PolynomialKernel, :SigmoidKernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{ScalarProductKernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
        function convert{T<:FloatingPoint}(::Type{StandardKernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
        function convert{T<:FloatingPoint}(::Type{SimpleKernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
        function convert{T<:FloatingPoint}(::Type{Kernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
    end
end
