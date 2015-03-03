#===================================================================================================
  Standard Kernel Functions
===================================================================================================#

function show(io::IO, κ::StandardKernel)
    print(io, " " * description_string(κ))
end


#=================================================
  Stationary Kernels
=================================================#

abstract StationaryKernel <: StandardKernel

is_euclidean_distance(κ::StationaryKernel) = true

@inline function kernel_function{T<:FloatingPoint}(κ::StationaryKernel, x::Vector{T}, y::Vector{T})
    kernel_function(κ, euclidean_distance(x, y))
end


#== Gaussian Kernel ===============#

type GaussianKernel <: StationaryKernel
    η::Real
    function GaussianKernel(η::Real=1)
        η > 0 || error("σ = $(η) must be greater than 0.")
        new(η)
    end
end

@inline gaussian_kernel{T<:FloatingPoint}(ϵᵗϵ::T, η::T) = exp(-η*ϵᵗϵ)
@inline function gaussian_kernel{T<:FloatingPoint}(ϵᵗϵ::T, κ::GaussianKernel)
    gaussian_kernel(ϵᵗϵ, T(κ.η))
end

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

type LaplacianKernel <: StationaryKernel
    η::Real
    function LaplacianKernel(η::Real=1)
        η > 0 || error("η = $(η) must be greater than zero.")
        new(η)
    end
end

@inline laplacian_kernel{T<:FloatingPoint}(ϵᵗϵ::T, η::T) = exp(-η*sqrt(ϵᵗϵ))
@inline function laplacian_kernel{T<:FloatingPoint}(ϵᵗϵ::T, κ::LaplacianKernel)
    laplacian_kernel(ϵᵗϵ, T(κ.η))
end

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

type RationalQuadraticKernel <: StationaryKernel
    c::Real
    function RationalQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

@inline rational_quadratic_kernel{T<:FloatingPoint}(ϵᵗϵ::T, c::T) = one(T) - ϵᵗϵ/(ϵᵗϵ + c)
@inline function rational_quadratic_kernel{T<:FloatingPoint}(ϵᵗϵ::T, κ::RationalQuadraticKernel)
    rational_quadratic_kernel(ϵᵗϵ, T(κ.c))
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

type MultiQuadraticKernel <: StationaryKernel
    c::Real
    function MultiQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

@inline multiquadratic_kernel{T<:FloatingPoint}(ϵᵗϵ::T, c::T) = sqrt(ϵᵗϵ + c)
@inline function multiquadratic_kernel{T<:FloatingPoint}(ϵᵗϵ::T, κ::MultiQuadraticKernel)
    multiquadratic_kernel(ϵᵗϵ, T(κ.c))
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

type InverseMultiQuadraticKernel <: StandardKernel
    c::Real
    function InverseMultiQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

@inline function inverse_multiquadratic_kernel{T<:FloatingPoint}(ϵᵗϵ::T, c::T) 
    one(T) / sqrt(ϵᵗϵ + c)
end
@inline function inverse_multiquadratic_kernel{T<:FloatingPoint}(ϵᵗϵ::T, 
                                                                 κ::InverseMultiQuadraticKernel)
    inverse_multiquadratic_kernel(ϵᵗϵ, T(κ.c))
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

type PowerKernel <: StationaryKernel
    d::Real
    function PowerKernel(d::Real = 1)
        d > 0 || error("d = $(d) must be greater than zero.")
        new(d)
    end
end

@inline power_kernel{T<:FloatingPoint}(ϵᵗϵ::T, d::T) = -(ϵᵗϵ^d)
@inline power_kernel{T<:FloatingPoint}(ϵᵗϵ::T, κ::PowerKernel) = power_kernel(ϵᵗϵ, T(κ.d))

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

type LogKernel <: StationaryKernel
    d::Real
    function LogKernel(d::Real = 1)
        d > 0 || error("d = $(d) must be greater than zero.")
        new(d)
    end
end

@inline log_kernel{T<:FloatingPoint}(ϵᵗϵ::T, d::T) = -log(ϵᵗϵ^d + one(T))
@inline log_kernel{T<:FloatingPoint}(ϵᵗϵ::T, κ::LogKernel) = log_kernel(ϵᵗϵ, T(κ.d))

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


#=================================================
  Non-Stationary Kernels
=================================================#

abstract NonStationaryKernel <: StandardKernel

is_scalar_product(κ::NonStationaryKernel) = true

@inline function kernel_function{T<:FloatingPoint}(κ::NonStationaryKernel, x::Vector{T},
                                                   y::Vector{T})
    kernel_function(κ, scalar_product(x, y))
end


#== Linear Kernel ====================#

type LinearKernel <: NonStationaryKernel
    c::Real
    function LinearKernel(c::Real=0)
        c >= 0 || error("c = $c must be greater than zero.")
        new(c)
    end
end

@inline linear_kernel{T<:FloatingPoint}(xᵗy::T, c::T) = xᵗy + c
@inline function linear_kernel{T<:FloatingPoint}(xᵗy::T, κ::LinearKernel)
    linear_kernel(xᵗy, T(κ.c))
end

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

type PolynomialKernel <: NonStationaryKernel
    α::Real
    c::Real
    d::Real
    function PolynomialKernel(α::Real=1, c::Real=1, d::Real=2)
        α > 0 || error("α = $(α) must be greater than zero.")
        c >= 0 || error("c = $(c) must be a non-negative number.")
        d > 0 || error("d = $(d) must be greater than zero.") 
        new(α, c, d)
    end
end

@inline polynomial_kernel{T<:FloatingPoint}(xᵗy::T, α::T, c::T, d::T) = (α*xᵗy + c)^d
@inline function polynomial_kernel{T<:FloatingPoint}(xᵗy::T, κ::PolynomialKernel)
    polynomial_kernel(xᵗy, T(κ.α), T(κ.c), T(κ.d))
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

type SigmoidKernel <: NonStationaryKernel
    α::Real
    c::Real
    function SigmoidKernel(α::Real=1, c::Real=0)
        α > 0 || error("α = $(α) must be greater than zero.")
        c >= 0 || error("c = $(c) must be non-negative.")
        new(α, c)
    end
end

@inline sigmoid_kernel{T<:FloatingPoint}(xᵗy::T, α::T, c::T) = tanh(α*xᵗy + c)
@inline function sigmoid_kernel{T<:FloatingPoint}(xᵗy::T, κ::SigmoidKernel)
    sigmoid_kernel(xᵗy, T(κ.α), T(κ.c))
end

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


#===================================================================================================
  Definitions (until return typed generic functions are optimised
===================================================================================================#

for (kernel, kf) in ((:GaussianKernel, :gaussian_kernel),
                     (:LaplacianKernel, :laplacian_kernel),
                     (:RationalQuadraticKernel, :rational_quadratic_kernel),
                     (:MultiQuadraticKernel, :multiquadratic_kernel),
                     (:InverseMultiQuadraticKernel, :inverse_multiquadratic_kernel),
                     (:PowerKernel, :power_kernel),
                     (:LogKernel, :log_kernel),
                     (:LinearKernel, :linear_kernel),
                     (:PolynomialKernel, :polynomial_kernel),
                     (:SigmoidKernel, :sigmoid_kernel))
    @eval begin
        @inline kernel_function{T<:FloatingPoint}(κ::$(kernel), xᵗy::T) = $(kf)(xᵗy, κ)
        function kernel_function!{T<:FloatingPoint}(κ::$(kernel), G::Array{T})
            args = map(T, arguments(κ))
            @inbounds for i = 1:length(G)
                G[i] = $(kf)(G[i], args...)
            end
            G
        end
        kernel_function{T<:FloatingPoint}(κ::$(kernel), G::Array{T}) = kernel_function(κ, G)
    end
end
