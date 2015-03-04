#===================================================================================================
  Standard Kernel Functions
===================================================================================================#

function show(io::IO, κ::StandardKernel)
    print(io, " " * description_string(κ))
end

is_euclidean_distance(κ::StandardKernel) = false
is_scalar_product(κ::StandardKernel) = false

function kernel_function!{T<:FloatingPoint}(κ::StandardKernel{T}, G::Array{T})
    @inbounds for i = 1:length(G)
        G[i] = kernel_function(κ, G[i])
    end
    G
end
kernel_function{T<:FloatingPoint}(κ::StandardKernel{T}, G::Array{T}) = kernel_function(κ, G)


#===========================================================================
  Auxiliary Functions
===========================================================================#

# xᵗy
function scalar_product{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    n = length(x)
    BLAS.dot(n, x, 1, y, 1)
end

# ϵᵗϵ = (x-y)ᵗ(x-y)
function euclidean_distance{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    n = length(x)
    ϵ = BLAS.axpy!(n, -one(T), y, 1, copy(x), 1)
    BLAS.dot(n, ϵ, 1, ϵ, 1)
end


#==========================================================================
  Stationary Kernels
==========================================================================#

abstract StationaryKernel{T<:FloatingPoint} <: StandardKernel{T}

is_euclidean_distance(κ::StationaryKernel) = true

@inline function kernel_function{T<:FloatingPoint}(κ::StationaryKernel{T}, x::Vector{T}, 
                                                   y::Vector{T})
    kernel_function(κ, euclidean_distance(x, y))
end


#== Gaussian Kernel ===============#

immutable GaussianKernel{T<:FloatingPoint} <: StationaryKernel{T}
    η::T
    function GaussianKernel(η::T)
        η > 0 || error("σ = $(η) must be greater than 0.")
        new(η)
    end
end
GaussianKernel{T<:FloatingPoint}(η::T = 1.0) = GaussianKernel{T}(η)

@inline kernel_function{T<:FloatingPoint}(κ::GaussianKernel{T}, ϵᵗϵ::T) =  exp(-κ.η*ϵᵗϵ)

arguments(κ::GaussianKernel) = (κ.η,)
isposdef_kernel(κ::GaussianKernel) = true

formula_string(κ::GaussianKernel) = "exp(-η‖x-y‖²)"
argument_string(κ::GaussianKernel) = "η = $(κ.η)"
description_string(κ::GaussianKernel) = "GaussianKernel(η=$(κ.η))"

function description(κ::GaussianKernel)
    print(
        """ 
         Gaussian Kernel:
         ===================================================================
         The Gaussian kernel is a radial basis function based on the
         Gaussian distribution's probability density function. The feature
         has an infinite number of dimensions.

             k(x,y) = exp(-η‖x-y‖²)    x ∈ ℝⁿ, y ∈ ℝⁿ, η > 0

         Since the value of the function decreases as x and y differ, it can
         be interpretted as a similarity measure.
        """
    )
end


#== Laplacian Kernel ===============#

type LaplacianKernel{T<:FloatingPoint} <: StationaryKernel{T}
    η::T
    function LaplacianKernel(η::T)
        η > 0 || error("η = $(η) must be greater than zero.")
        new(η)
    end
end
LaplacianKernel{T<:FloatingPoint}(η::T = 1.0) = LaplacianKernel{T}(η)

@inline kernel_function{T<:FloatingPoint}(κ::LaplacianKernel{T}, ϵᵗϵ::T) = exp(-κ.η*sqrt(ϵᵗϵ))

arguments(κ::LaplacianKernel) = (κ.η,)
isposdef_kernel(κ::LaplacianKernel) = true

formula_string(κ::LaplacianKernel) = "exp(-η‖x-y‖)"
argument_string(κ::LaplacianKernel) = "η = $(κ.η)"
description_string(κ::LaplacianKernel) = "LaplacianKernel(η=$(κ.η))"

function description(κ::LaplacianKernel)
    print(
        """ 
         Laplacian Kernel:
         ===================================================================
         The Laplacian (exponential) kernel is a radial basis function that
         differs from the Gaussian kernel in that it is a less sensitive
         similarity measure. Similarly, it is less sensitive to changes in
         the parameter η:

             k(x,y) = exp(-η‖x-y‖)    x ∈ ℝⁿ, y ∈ ℝⁿ, η > 0
        """
    )
end


#== Rational Quadratic Kernel ===============#

type RationalQuadraticKernel{T<:FloatingPoint} <: StationaryKernel{T}
    c::T
    function RationalQuadraticKernel(c::T)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end
RationalQuadraticKernel{T<:FloatingPoint}(c::T = 1.0) = RationalQuadraticKernel{T}(c)

@inline function kernel_function{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, ϵᵗϵ::T)
    one(T) - ϵᵗϵ/(ϵᵗϵ + κ.c)
end

arguments(κ::RationalQuadraticKernel) = (κ.c,)
isposdef_kernel(κ::RationalQuadraticKernel) = true

formula_string(κ::RationalQuadraticKernel) = "1 - ‖x-y‖²/(‖x-y‖² + c)"
argument_string(κ::RationalQuadraticKernel) = "c = $(κ.c)"
description_string(κ::RationalQuadraticKernel) = "RationalQuadraticKernel(c=$(κ.c))"

function description(κ::RationalQuadraticKernel)
    print(
        """ 
         Rational Quadratic Kernel:
         ===================================================================
         The rational quadratic kernel is a stationary kernel that is
         similar in shape to the Gaussian kernel:

             k(x,y) = 1 - ‖x-y‖²/(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
        """
    )
end


#== Multi-Quadratic Kernel ===============#

immutable MultiQuadraticKernel{T<:FloatingPoint} <: StationaryKernel{T}
    c::T
    function MultiQuadraticKernel(c::T)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end
MultiQuadraticKernel{T<:FloatingPoint}(c::T = 1.0) = MultiQuadraticKernel{T}(c)

@inline function kernel_function{T<:FloatingPoint}(κ::MultiQuadraticKernel{T}, ϵᵗϵ::T)
    sqrt(ϵᵗϵ + κ.c)
end

arguments(κ::MultiQuadraticKernel) = (κ.c,)
isposdef_kernel(κ::MultiQuadraticKernel) = false

formula_string(κ::MultiQuadraticKernel) = "√(‖x-y‖² + c)"
argument_string(κ::MultiQuadraticKernel) = "c = $(κ.c)"
description_string(κ::MultiQuadraticKernel) = "MultiQuadraticKernel(c=$(κ.c))"

function description(κ::MultiQuadraticKernel)
    print(
        """ 
         Multi-Quadratic Kernel:
         ===================================================================
         The multi-quadratic kernel is a positive semidefinite kernel:

             k(x,y) = √(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
        """
    )
end


#== Inverse Multi-Quadratic Kernel ===============#

immutable InverseMultiQuadraticKernel{T<:FloatingPoint} <: StandardKernel{T}
    c::T
    function InverseMultiQuadraticKernel(c::T)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end
InverseMultiQuadraticKernel{T<:FloatingPoint}(c::T = 1.0) = InverseMultiQuadraticKernel{T}(c)

@inline function kernel_function{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T}, ϵᵗϵ::T)
    one(T) / sqrt(ϵᵗϵ + κ.c)
end

arguments(κ::InverseMultiQuadraticKernel) = (κ.c,)
isposdef_kernel(κ::InverseMultiQuadraticKernel) = false

formula_string(κ::InverseMultiQuadraticKernel) = "1/√(‖x-y‖² + c)"
argument_string(κ::InverseMultiQuadraticKernel) = "c = $(κ.c)"
description_string(κ::InverseMultiQuadraticKernel) = "InverseMultiQuadraticKernel(c=$(κ.c))"

function description(κ::InverseMultiQuadraticKernel)
    print(
        """ 
         Inverse Multi-Quadratic Kernel:
         ===================================================================
         The inverse multi-quadratic kernel is a radial basis function. The
         resulting feature has an infinite number of dimensions:

             k(x,y) = 1/√(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
        """
    )
end


#== Power Kernel ===============#

immutable PowerKernel{T<:FloatingPoint} <: StationaryKernel{T}
    d::T
    function PowerKernel(d::T)
        d > 0 || error("d = $(d) must be greater than zero.")
        new(d)
    end
end
PowerKernel{T<:FloatingPoint}(d::T) = PowerKernel{T}(d)

@inline kernel_function{T<:FloatingPoint}(κ::PowerKernel{T}, ϵᵗϵ::T) = -ϵᵗϵ^(κ.d)

arguments(κ::PowerKernel) = (κ.d,)
isposdef_kernel(κ::PowerKernel) = false

formula_string(κ::PowerKernel) = "-‖x-y‖ᵈ"
argument_string(κ::PowerKernel) = "d = $(κ.d)"
description_string(κ::PowerKernel) = "PowerKernel(d=$(κ.d))"

function description(κ::PowerKernel)
    print(
        """ 
         Power Kernel:
         ===================================================================
         The power kernel (also known as the unrectified triangular kernel)
         is a positive semidefinite kernel. An important feature of the
         power kernel is that it is scale invariant. The function is given
         by:

             k(x,y) = -‖x-y‖ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, d > 0
        """
    )
end


#== Log Kernel ===============#

immutable LogKernel{T<:FloatingPoint} <: StationaryKernel{T}
    d::T
    function LogKernel(d::T)
        d > 0 || error("d = $(d) must be greater than zero.")
        new(d)
    end
end
LogKernel{T<:FloatingPoint}(d::T) = LogKernel{T}(d)

@inline kernel_function{T<:FloatingPoint}(ϵᵗϵ::T, κ::LogKernel{T}) = -log(ϵᵗϵ^(κ.d) + one(T))

arguments(κ::LogKernel) = (κ.d,)
isposdef_kernel(κ::LogKernel) = false

formula_string(κ::LogKernel) = "-‖x-y‖ᵈ"
argument_string(κ::LogKernel) = "d = $(κ.d)"
description_string(κ::LogKernel) = "LogKernel(d=$(κ.d))"

function description(κ::LogKernel)
    print(
        """ 
         Log Kernel:
         ===================================================================
         The power kernel is a positive semidefinite kernel. The function is
         given by:

             k(x,y) = -log(‖x-y‖ᵈ + 1)    x ∈ ℝⁿ, y ∈ ℝⁿ, d > 0
        """
    )
end


#==========================================================================
  Non-Stationary Kernels
==========================================================================#

abstract NonStationaryKernel{T<:FloatingPoint} <: StandardKernel{T}

is_scalar_product(κ::NonStationaryKernel) = true

@inline function kernel_function{T<:FloatingPoint}(κ::NonStationaryKernel{T}, x::Vector{T},
                                                   y::Vector{T})
    kernel_function(κ, scalar_product(x, y))
end


#== Linear Kernel ====================#

immutable LinearKernel{T<:FloatingPoint} <: NonStationaryKernel{T}
    c::T
    function LinearKernel(c::T)
        c >= 0 || error("c = $c must be greater than zero.")
        new(c)
    end
end
LinearKernel{T<:FloatingPoint}(c::T = 1.0) = LinearKernel{T}(c)

@inline kernel_function{T<:FloatingPoint}(κ::LinearKernel, xᵗy::T) = xᵗy + κ.c

arguments(κ::LinearKernel) = (κ.c,)
isposdef_kernel(κ::LinearKernel) = true

formula_string(κ::LinearKernel) = "k(x,y) = xᵗy + c"
argument_string(κ::LinearKernel) = "c = $(κ.c)"
description_string(κ::LinearKernel) = "LinearKernel(c=$(κ.c))"

function description(κ::LinearKernel)
    print(
        """ 
         Linear Kernel:
         ===================================================================
         The linear kernel differs from the ordinary inner product by the
         addition of an optional constant c ≥ 0:

             k(x,y) = xᵗy + c    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0

         Techniques using the linear kernel often do not differ from their
         non-kernelized versions.
        """
    )
end


#== Polynomial Kernel ===============#

immutable PolynomialKernel{T<:FloatingPoint} <: NonStationaryKernel{T}
    α::T
    c::T
    d::T
    function PolynomialKernel(α::T, c::T, d::T)
        α > 0 || error("α = $(α) must be greater than zero.")
        c >= 0 || error("c = $(c) must be a non-negative number.")
        d > 0 || error("d = $(d) must be greater than zero.") 
        new(α, c, d)
    end
end
function PolynomialKernel{T<:FloatingPoint}(α::T = 1.0, c::T = one(T), d::T = T(2))
    PolynomialKernel{T}(α, c, d)
end

@inline function kernel_function{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵗy::T)
    (κ.α*xᵗy + κ.c)^κ.d
end

arguments(κ::PolynomialKernel) = (κ.α, κ.c, κ.d)
isposdef_kernel(κ::PolynomialKernel) = true

formula_string(κ::PolynomialKernel) = "(αxᵗy + c)ᵈ"
argument_string(κ::PolynomialKernel) = "α = $(κ.α), c = $(κ.c) and d = $(κ.d)"
description_string(κ::PolynomialKernel) = "PolynomialKernel(α=$(κ.α),c=$(κ.c),d=$(κ.d))"

function description(κ::PolynomialKernel)
    print(
        """ 
         Polynomial Kernel:
         ===================================================================
         The polynomial kernel is a non-stationary kernel which represents
         the original features as in a feature space over polynomials up to 
         degree d of the original variables:

             k(x,y) = (αxᵗy + c)ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0, d > 0

         This kernel is sensitive to numerical instability in the case that
         d is increasingly large and αxᵗy + c approaches zero.
        """
    )
end


#== Sigmoid Kernel ===============#

immutable SigmoidKernel{T<:FloatingPoint} <: NonStationaryKernel{T}
    α::T
    c::T
    function SigmoidKernel(α::T, c::T)
        α > 0 || error("α = $(α) must be greater than zero.")
        c >= 0 || error("c = $(c) must be non-negative.")
        new(α, c)
    end
end
SigmoidKernel{T<:FloatingPoint}(α::T = 1.0, c::T = one(T)) = SigmoidKernel{T}(α, c)

@inline kernel_function{T<:FloatingPoint}(κ::SigmoidKernel, xᵗy::T) = tanh(κ.α*xᵗy + κ.c)

arguments(κ::SigmoidKernel) = (κ.α, κ.c)
isposdef_kernel(κ::SigmoidKernel) = false

formula_string(κ::SigmoidKernel) = "tanh(α‖x-y‖² + c)"
argument_string(κ::SigmoidKernel) = "α = $(κ.α) and c = $(κ.c)"
description_string(κ::SigmoidKernel) = "SigmoidKernel(α=$(κ.α),c=$(κ.c))"

function description(κ::SigmoidKernel)
    print(
        """ 
         Sigmoid Kernel:
         ===================================================================
         The sigmoid kernel is only positive semidefinite. It is used in the
         field of neural networks where it is often used as the activation
         function for artificial neurons.

             k(x,y) = tanh(αxᵗy + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0
        """
    )
end


#==========================================================================
  Pointwise Product Kernels
==========================================================================#

#=
type PointwiseProductKernel <: StandardKernel
    f::Function
    posdef::Bool
    function PointwiseProductKernel(f::Function, posdef::Bool = false)
        method_exists(f, (Array{Float32},)) && method_exists(f, (Array{Float64},)) || (
            error("f = $(f) must map f: ℝⁿ → ℝ (define methods for both Array{Float32} and " * ( 
                  "Array{Float64}).")))
        new(f, posdef)
    end
end

@inline function pointwise_product_kernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, f::Function)
    f(x) * f(y)
end
@inline function pointwise_product_kernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, 
                                                            κ::PointwiseProductKernel)
    pointwise_product_kernel(x, y, κ.f)
end

arguments(κ::PointwiseProductKernel) = (κ.f,)
isposdef_kernel(κ::PointwiseProductKernel) = κ.posdef

@inline function kernel_function(κ::PointwiseProductKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = pointwiseproductkernel(x, y, copy(κ.f))
end

formula_string(κ::PointwiseProductKernel) = "k(x,y) = f(x)f(y)"
argument_string(κ::PointwiseProductKernel) = "f = $(κ.f)"
description_string(κ::PointwiseProductKernel) = "PointwiseProductKernel(f=$(κ.f))"

function description(κ::PointwiseProductKernel)
    print(
        """ 
         Pointwise Product Kernel:
         ===================================================================
         The pointwise product kernel is the product of a real-valued multi-
         variate function applied to each of the vector arguments:

             k(x,y) = f(x)f(y)    x ∈ ℝⁿ, y ∈ ℝⁿ, f: ℝⁿ → ℝ
        """
    )
end
=#

#=================================================
  Generic Kernels
=================================================#

#=
type GenericKernel <: StandardKernel
    k::Function
    posdef::Bool
    function GenericKernel(k::Function, posdef::Bool = false)
        method_exists(f, (Array{Float32}, Array{Float32})) && (
            method_exists(f, (Array{Float64}, Array{Float64})) || (
            error("k = $(f) must map k: ℝⁿ×ℝⁿ → ℝ (define methods for both" * (
                  "Array{Float32} and Array{Float64})."))))
        new(k, posdef)
    end
end

arguments(κ::GenericKernel) = (κ.k,)
isposdef_kernel(κ::GenericKernel) = κ.posdef

kernel_function(κ::GenericKernel) = copy(κ.k)

formula_string(κ::GenericKernel) = "k(x,y)"
argument_string(κ::GenericKernel) = "k = $(κ.k)"
description_string(κ::GenericKernel) = "GenericKernel(k=$(κ.k))"

function description(κ::GenericKernel)
    print(
        """ 
         Generic Kernel:
         ===================================================================
         Customized definition:

             k: ℝⁿ×ℝⁿ → ℝ
        """
    )
end
=#
