function show(io::IO, κ::StandardKernel)
    print(io, " " * description_string(κ))
end

#=================================================
  Stationary Kernels
=================================================#

abstract StationaryKernel <: StandardKernel

is_euclidian_distance(κ::StationaryKernel) = true

@inline function kernel_function(κ::StationaryKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = (
        scalar_kernel_function(κ)(euclidean_distance(x, y)))
end


#== Gaussian Kernel ===============#

@inline gaussian_kernel{T<:FloatingPoint}(ϵᵗϵ::T, η::T) = exp(-η*ϵᵗϵ)
function gaussian_kernel!{T<:FloatingPoint}(G::Array{T}, η::T)
    @inbounds for i = 1:length(G)
        G[i] = exp(-η*G[i])
    end
    G
end
gaussian_kernel{T<:FloatingPoint}(G::Matrix{T}, η::T) = gaussian_kernel!(copy(G), η)

type GaussianKernel <: StationaryKernel
    η::Real
    function GaussianKernel(η::Real=1)
        η > 0 || error("σ = $(η) must be greater than 0.")
        new(η)
    end
end

arguments(κ::GaussianKernel) = (κ.η,)
isposdef_kernel(κ::GaussianKernel) = true

function scalar_kernel_function(κ::GaussianKernel)
    k{T<:FloatingPoint}(ϵᵗϵ::T) = gaussian_kernel(ϵᵗϵ, T(κ.η))
end

function vectorized_kernel_function!(κ::GaussianKernel)
    k{T<:FloatingPoint}(G::Array{T}) = gaussian_kernel!(G, T(κ.η))
end

function vectorized_kernel_function(κ::GaussianKernel)
    k{T<:FloatingPoint}(G::Array{T}) = gaussian_kernel(G, T(κ.η))
end

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

@inline laplacian_kernel{T<:FloatingPoint}(ϵᵗϵ::T, η::T) = exp(-η*sqrt(ϵᵗϵ))
function laplacian_kernel!{T<:FloatingPoint}(G::Array{T}, η::T)
    @inbounds for i = 1:length(G)
        G[i] = exp(-η*sqrt(G[i]))
    end
    G
end
laplacian_kernel{T<:FloatingPoint}(G::Array{T}, η::T) = laplacian_kernel(copy(G), η)

type LaplacianKernel <: StationaryKernel
    η::Real
    function LaplacianKernel(η::Real=1)
        η > 0 || error("η = $(η) must be greater than zero.")
        new(η)
    end
end

arguments(κ::LaplacianKernel) = (κ.η,)
isposdef_kernel(κ::LaplacianKernel) = true

@inline function scalar_kernel_function(κ::LaplacianKernel)
    k{T<:FloatingPoint}(ϵᵗϵ::T) = laplacian_kernel(ϵᵗϵ, T(κ.η))
end

@inline function vectorized_kernel_function!(κ::LaplacianKernel)
    k{T<:FloatingPoint}(G::Array{T}) = laplacian_kernel!(G, T(κ.η))
end

@inline function vectorized_kernel_function(κ::LaplacianKernel)
    k{T<:FloatingPoint}(G::Array{T}) = laplacian_kernel(G, T(κ.η))
end

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

@inline rational_quadratic_kernel{T<:FloatingPoint}(ϵᵗϵ::T, c::T) = one(T) - ϵᵗϵ/(ϵᵗϵ + c)
function rational_quadratic_kernel!{T<:FloatingPoint}(G::Array{T}, c::T)
    @inbounds for i = 1:length(G)
        G[i] = one(T) - G[i]/(G[i] + c)
    end
    G
end
@inline function rational_quadratic_kernel{T<:FloatingPoint}(G::Array{T}, c::T)
    rational_quadratic_kernel!(copy(G), c)
end

type RationalQuadraticKernel <: StationaryKernel
    c::Real
    function RationalQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

arguments(κ::RationalQuadraticKernel) = (κ.c,)
isposdef_kernel(κ::RationalQuadraticKernel) = true

@inline function scalar_kernel_function(κ::RationalQuadraticKernel)
    k{T<:FloatingPoint}(ϵᵗϵ::T) = rational_quadratic_kernel(ϵᵗϵ, κ.c)
end

@inline function vectorized_kernel_function!(κ::RationalQuadraticKernel)
    k{T<:FloatingPoint}(G::Array{T}) = rational_quadratic_kernel!(G, T(κ.c))
end

@inline function vectorized_kernel_function(κ::RationalQuadraticKernel)
    k{T<:FloatingPoint}(G::Array{T}) = rational_quadratic_kernel(G, T(κ.c))
end

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

@inline function multiquadratic_kernel{T<:FloatingPoint}(ϵᵗϵ::T, c::Real)
    sqrt(ϵᵗϵ + convert(T,c))
end

type MultiQuadraticKernel <: StationaryKernel
    c::Real
    function MultiQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

arguments(κ::MultiQuadraticKernel) = (κ.c,)
isposdef_kernel(κ::MultiQuadraticKernel) = false

@inline function kernel_function(κ::MultiQuadraticKernel)
    k{T<:FloatingPoint}(ϵᵗϵ::T) = multiquadratic_kernel(ϵᵗϵ, κ.c)
end

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

@inline function inverse_multiquadratic_kernel{T<:FloatingPoint}(ϵᵗϵ::T, c::Real)
    one(T) / sqrt(ϵᵗϵ + convert(T, c))
end

type InverseMultiQuadraticKernel <: StandardKernel
    c::Real
    function InverseMultiQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

arguments(κ::InverseMultiQuadraticKernel) = κ.c
isposdef_kernel(κ::InverseMultiQuadraticKernel) = false

@inline function scalar_kernel_function(κ::InverseMultiQuadraticKernel)
    k{T<:FloatingPoint}(ϵᵗϵ::T) = inverse_multiquadratic_kernel(ϵᵗϵ, κ.c)
end

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

@inline power_kernel{T<:FloatingPoint}(ϵᵗϵ::T, d::T) = -(ϵᵗϵ^d)
function power_kernel!{T<:FloatingPoint}(G::Array{T}, d::T)
    @inbounds for i = 1:length(G)
        G[i] = -(G[i]^d)
    end
    G
end
power_kernel{T<:FloatingPoint}(G::Array{T}, d::T) = power_kernel!(copy(G), d)

type PowerKernel <: StationaryKernel
    d::Real
    function PowerKernel(d::Real = 1)
        d > 0 || error("d = $(d) must be greater than zero.")
        new(d)
    end
end

arguments(κ::PowerKernel) = (κ.d,)
isposdef_kernel(κ::PowerKernel) = false

@inline function scalar_kernel_function(κ::PowerKernel)
    k{T<:FloatingPoint}(ϵᵗϵ::T) = power_kernel(ϵᵗϵ, κ.d)
end

@inline function vectorized_kernel_function!(κ::PowerKernel)
    k{T<:FloatingPoint}(G::Array{T}) = power_kernel!(G, T(κ.d))
end

@inline function vectorized_kernel_function(κ::PowerKernel)
    k{T<:FloatingPoint}(G::Array{T}) = power_kernel(G, T(κ.d))
end

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

@inline log_kernel{T<:FloatingPoint}(ϵᵗϵ::T, d::Real) = -log(ϵᵗϵ^convert(T,d) + one(T))

type LogKernel <: StationaryKernel
    d::Real
    function LogKernel(d::Real = 1)
        d > 0 || error("d = $(d) must be greater than zero.")
        new(d)
    end
end

arguments(κ::LogKernel) = (κ.d,)
isposdef_kernel(κ::LogKernel) = false

@inline function scalar_kernel_function(κ::LogKernel)
    k{T<:FloatingPoint}(ϵᵗϵ::T) = log_kernel(ϵᵗϵ, κ.d)
end

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

@inline function kernel_function(κ::NonStationaryKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = (
        scalar_kernel_function(κ)(scalar_product(x, y)))
end


#== Linear Kernel ====================#

@inline linear_kernel{T<:FloatingPoint}(xᵗy::T, c::T) = xᵗy + c
function linear_kernel!{T<:FloatingPoint}(G::Array{T}, c::T)
    @inbounds for i = 1:length(G)
        G[i] = G[i] + c
    end
    G
end
linear_kernel{T<:FloatingPoint}(G::Array{T}, c::T) = linear_kernel!(copy(G), c)

type LinearKernel <: NonStationaryKernel
    c::Real
    function LinearKernel(c::Real=0)
        c >= 0 || error("c = $c must be greater than zero.")
        new(c)
    end
end

arguments(κ::LinearKernel) = (κ.c,)
isposdef_kernel(κ::LinearKernel) = true

@inline function scalar_kernel_function(κ::LinearKernel)
    k{T<:FloatingPoint}(xᵗy::T) = linearkernel(xᵗy, κ.c)
end

@inline function vectorized_kernel_function!(κ::LinearKernel)
    k{T<:FloatingPoint}(G::Matrix{T}) = linear_kernel!(G, κ.c)
end

@inline function vectorized_kernel_function(κ::LinearKernel)
    k{T<:FloatingPoint}(G::Matrix{T}) = linear_kernel(G, κ.c)
end

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

@inline polynomial_kernel{T<:FloatingPoint}(xᵗy::T, α::T, c::T, d::T) = (α*xᵗy + c)^d
function polynomial_kernel!{T<:FloatingPoint}(G::Array{T}, α::T, c::T, d::T)
    @inbounds for i = 1:length(G)
        G[i] = (α*G[i] + c)^d
    end
    G
end
polynomial_kernel{T<:FloatingPoint}(G::Array{T}, α::T, c::T, d::T) = polynomial_kernel!(copy(G), α, c, d)

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

arguments(κ::PolynomialKernel) = (κ.α, κ.c, κ.d)
isposdef_kernel(κ::PolynomialKernel) = true

@inline function scalar_kernel_function(κ::PolynomialKernel)
    k{T<:FloatingPoint}(xᵗy::T) = polynomial_kernel(xᵗy, T(κ.α), T(κ.c), T(κ.d))
end

@inline function vectorized_kernel_function!(κ::PolynomialKernel)
    k!{T<:FloatingPoint}(G::Matrix{T}) = polynomial_kernel!(G, T(κ.α), T(κ.c), T(κ.d))
end

@inline function vectorized_kernel_function!(κ::PolynomialKernel)
    k!{T<:FloatingPoint}(G::Matrix{T}) = polynomial_kernel(G, T(κ.α), T(κ.c), T(κ.d))
end

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

@inline sigmoid_kernel{T<:FloatingPoint}(xᵗy::T, α::T, c::T) = tanh(α*xᵗy + c)
function sigmoid_kernel!{T<:FloatingPoint}(G::Array{T}, α::T, c::T)
    @inbounds for i = 1:length(G)
        G[i] = tanh(α*G[i] + c)
    end
    G
end
sigmoid_kernel{T<:FloatingPoint}(G::Array{T}, α::T, c::T) = sigmoid_kernel!(copy(G), α, c)

type SigmoidKernel <: NonStationaryKernel
    α::Real
    c::Real
    function SigmoidKernel(α::Real=1, c::Real=0)
        α > 0 || error("α = $(α) must be greater than zero.")
        c >= 0 || error("c = $(c) must be non-negative.")
        new(α, c)
    end
end

arguments(κ::SigmoidKernel) = (κ.α, κ.c)
isposdef_kernel(κ::SigmoidKernel) = false

@inline function scalar_kernel_function!(κ::SigmoidKernel)
    k{T<:FloatingPoint}(xᵗy::T) = sigmoid_kernel(xᵗy, κ.α, κ.c)
end

@inline function vectorized_kernel_function!(κ::SigmoidKernel)
    k{T<:FloatingPoint}(G::Array{T}) = sigmoid_kernel!(G, T(κ.α), T(κ.c))
end

@inline function vectorized_kernel_function(κ::SigmoidKernel)
    k{T<:FloatingPoint}(G::Array{T}) = sigmoid_kernel(G, T(κ.α), T(κ.c))
end

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


