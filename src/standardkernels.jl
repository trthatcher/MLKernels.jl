#===================================================================================================
  Standard Kernel Functions
===================================================================================================#

abstract StandardKernel{T<:FloatingPoint} <: SimpleKernel{T}

function show(io::IO, κ::StandardKernel)
    print(io, description_string(κ))
end


#===========================================================================
  Auxiliary Functions
===========================================================================#

# xᵀy
function scalar_product{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    n = length(x)
    BLAS.dot(n, x, 1, y, 1)
end

# ϵᵀϵ = (x-y)ᵀ(x-y)
function euclidean_distance{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    n = length(x)
    ϵ = BLAS.axpy!(n, -one(T), y, 1, copy(x), 1)
    BLAS.dot(n, ϵ, 1, ϵ, 1)
end


#==========================================================================
  Euclidean Distance Kernels
==========================================================================#

abstract EuclideanDistanceKernel{T<:FloatingPoint} <: StandardKernel{T}

is_euclidean_distance(κ::EuclideanDistanceKernel) = true

function kernel_function{T<:FloatingPoint}(κ::EuclideanDistanceKernel{T}, x::Vector{T},
                                                   y::Vector{T})
    kernelize_scalar(κ, euclidean_distance(x, y))
end


#== Gaussian Kernel ===============#

immutable GaussianKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    η::T
    function GaussianKernel(η::T)
        η > 0 || throw(ArgumentError("σ = $(η) must be greater than 0."))
        new(η)
    end
end
GaussianKernel{T<:FloatingPoint}(η::T = 1.0) = GaussianKernel{T}(η)

function convert{T<:FloatingPoint}(::Type{GaussianKernel{T}}, κ::GaussianKernel) 
    GaussianKernel(convert(T, κ.η))
end

kernelize_scalar{T<:FloatingPoint}(κ::GaussianKernel{T}, ϵᵀϵ::T) = exp(-κ.η*ϵᵀϵ)

arguments(κ::GaussianKernel) = (κ.η,)
isposdef_kernel(κ::GaussianKernel) = true
is_stationary_kernel(κ::GaussianKernel) = true

formula_string(κ::GaussianKernel) = "exp(-η‖x-y‖²)"
argument_string(κ::GaussianKernel) = "η = $(κ.η)"
function description_string{T<:FloatingPoint}(κ::GaussianKernel{T}, eltype::Bool = true) 
    "GaussianKernel" * (eltype ? "{$(T)}" : "") * "(η=$(κ.η))"
end

function description(κ::GaussianKernel)
    print(
        """ 
         Gaussian Kernel:
         
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

type LaplacianKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    η::T
    function LaplacianKernel(η::T)
        η > 0 || throw(ArgumentError("η = $(η) must be greater than zero."))
        new(η)
    end
end
LaplacianKernel{T<:FloatingPoint}(η::T = 1.0) = LaplacianKernel{T}(η)

function convert{T<:FloatingPoint}(::Type{LaplacianKernel{T}}, κ::LaplacianKernel) 
    LaplacianKernel(convert(T, κ.η))
end

function kernelize_scalar{T<:FloatingPoint}(κ::LaplacianKernel{T}, ϵᵀϵ::T)
    exp(-κ.η*sqrt(ϵᵀϵ))
end

arguments(κ::LaplacianKernel) = (κ.η,)
isposdef_kernel(κ::LaplacianKernel) = true

formula_string(κ::LaplacianKernel) = "exp(-η‖x-y‖)"
argument_string(κ::LaplacianKernel) = "η = $(κ.η)"
function description_string{T<:FloatingPoint}(κ::LaplacianKernel{T}, eltype::Bool = true) 
    "LaplacianKernel" * (eltype ? "{$(T)}" : "") * "(η=$(κ.η))"
end

function description(κ::LaplacianKernel)
    print(
        """ 
         Laplacian Kernel:
         
         The Laplacian (exponential) kernel is a radial basis function that
         differs from the Gaussian kernel in that it is a less sensitive
         similarity measure. Similarly, it is less sensitive to changes in
         the parameter η:

             k(x,y) = exp(-η‖x-y‖)    x ∈ ℝⁿ, y ∈ ℝⁿ, η > 0
        """
    )
end


#== Rational Quadratic Kernel ===============#

type RationalQuadraticKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    c::T
    function RationalQuadraticKernel(c::T)
        c > 0 || throw(ArgumentError("c = $(c) must be greater than zero."))
        new(c)
    end
end
RationalQuadraticKernel{T<:FloatingPoint}(c::T = 1.0) = RationalQuadraticKernel{T}(c)

function convert{T<:FloatingPoint}(::Type{RationalQuadraticKernel{T}}, κ::RationalQuadraticKernel) 
    RationalQuadraticKernel(convert(T, κ.c))
end

function kernelize_scalar{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, ϵᵀϵ::T)
    one(T) - ϵᵀϵ/(ϵᵀϵ + κ.c)
end

arguments(κ::RationalQuadraticKernel) = (κ.c,)
isposdef_kernel(κ::RationalQuadraticKernel) = true

formula_string(κ::RationalQuadraticKernel) = "1 - ‖x-y‖²/(‖x-y‖² + c)"
argument_string(κ::RationalQuadraticKernel) = "c = $(κ.c)"
function description_string{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, eltype::Bool = true)
    "RationalQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(c=$(κ.c))"
end

function description(κ::RationalQuadraticKernel)
    print(
        """ 
         Rational Quadratic Kernel:
         
         The rational quadratic kernel is a stationary kernel that is
         similar in shape to the Gaussian kernel:

             k(x,y) = 1 - ‖x-y‖²/(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
        """
    )
end


#== Multi-Quadratic Kernel ===============#

immutable MultiQuadraticKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    c::T
    function MultiQuadraticKernel(c::T)
        c > 0 || throw(ArgumentError("c = $(c) must be greater than zero."))
        new(c)
    end
end
MultiQuadraticKernel{T<:FloatingPoint}(c::T = 1.0) = MultiQuadraticKernel{T}(c)

function convert{T<:FloatingPoint}(::Type{MultiQuadraticKernel{T}}, κ::MultiQuadraticKernel) 
    MultiQuadraticKernel(convert(T, κ.c))
end

function kernelize_scalar{T<:FloatingPoint}(κ::MultiQuadraticKernel{T}, ϵᵀϵ::T)
    sqrt(ϵᵀϵ + κ.c)
end

arguments(κ::MultiQuadraticKernel) = (κ.c,)
isposdef_kernel(κ::MultiQuadraticKernel) = false

formula_string(κ::MultiQuadraticKernel) = "√(‖x-y‖² + c)"
argument_string(κ::MultiQuadraticKernel) = "c = $(κ.c)"
function description_string{T<:FloatingPoint}(κ::MultiQuadraticKernel{T}, eltype::Bool = true)
    "MultiQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(c=$(κ.c))"
end

function description(κ::MultiQuadraticKernel)
    print(
        """ 
         Multi-Quadratic Kernel:
         
         The multi-quadratic kernel is a positive semidefinite kernel:

             k(x,y) = √(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
        """
    )
end


#== Inverse Multi-Quadratic Kernel ===============#

immutable InverseMultiQuadraticKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    c::T
    function InverseMultiQuadraticKernel(c::T)
        c > 0 || throw(ArgumentError("c = $(c) must be greater than zero."))
        new(c)
    end
end
InverseMultiQuadraticKernel{T<:FloatingPoint}(c::T = 1.0) = InverseMultiQuadraticKernel{T}(c)

function convert{T<:FloatingPoint}(::Type{InverseMultiQuadraticKernel{T}}, 
                                   κ::InverseMultiQuadraticKernel) 
    InverseMultiQuadraticKernel(convert(T, κ.c))
end

function kernelize_scalar{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T}, ϵᵀϵ::T)
    one(T) / sqrt(ϵᵀϵ + κ.c)
end

arguments(κ::InverseMultiQuadraticKernel) = (κ.c,)
isposdef_kernel(κ::InverseMultiQuadraticKernel) = false

formula_string(κ::InverseMultiQuadraticKernel) = "1/√(‖x-y‖² + c)"
argument_string(κ::InverseMultiQuadraticKernel) = "c = $(κ.c)"
function description_string{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T}, 
                                              eltype::Bool = true)
    "InverseMultiQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(c=$(κ.c))"
end

function description(κ::InverseMultiQuadraticKernel)
    print(
        """ 
         Inverse Multi-Quadratic Kernel:
         
         The inverse multi-quadratic kernel is a radial basis function. The
         resulting feature has an infinite number of dimensions:

             k(x,y) = 1/√(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
        """
    )
end


#== Power Kernel ===============#

immutable PowerKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    d::T
    function PowerKernel(d::T)
        d > 0 || throw(ArgumentError("d = $(d) must be a positive integer."))
        b = trunc(d)
        d == b || warn("d = $(d) was truncated to $(b).")
        new(b)
    end
end
PowerKernel{T<:FloatingPoint}(d::T = 2.0) = PowerKernel{T}(d)
PowerKernel(d::Integer) = PowerKernel(convert(Float64, d))

convert{T<:FloatingPoint}(::Type{PowerKernel{T}}, κ::PowerKernel) = PowerKernel(convert(T, κ.d))

kernelize_scalar{T<:FloatingPoint}(κ::PowerKernel{T}, ϵᵀϵ::T) = -ϵᵀϵ^(κ.d)

arguments(κ::PowerKernel) = (κ.d,)
isposdef_kernel(κ::PowerKernel) = false

formula_string(κ::PowerKernel) = "-‖x-y‖ᵈ"
argument_string(κ::PowerKernel) = "d = $(κ.d)"
function description_string{T<:FloatingPoint}(κ::PowerKernel{T}, eltype::Bool = true)
    "PowerKernel" * (eltype ? "{$(T)}" : "") * "(d=$(κ.d))"
end

function description(κ::PowerKernel)
    print(
        """ 
         Power Kernel:
         
         The power kernel (also known as the unrectified triangular kernel)
         is a positive semidefinite kernel. An important feature of the
         power kernel is that it is scale invariant. The function is given
         by:

             k(x,y) = -‖x-y‖ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, d > 0
        """
    )
end


#== Log Kernel ===============#

immutable LogKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    d::T
    function LogKernel(d::T)
        d > 0 || throw(ArgumentError("d = $(d) must be a positive integer."))
        b = trunc(d)
        d == b || warn("d = $(d) was truncated to $(b).")
        new(b)
    end
end
LogKernel{T<:FloatingPoint}(d::T = 1.0) = LogKernel{T}(d)
LogKernel(d::Integer) = LogKernel(convert(Float32, d))

convert{T<:FloatingPoint}(::Type{LogKernel{T}}, κ::LogKernel) = LogKernel(convert(T, κ.d))

function kernelize_scalar{T<:FloatingPoint}(κ::LogKernel{T}, ϵᵀϵ::T) 
    -log(sqrt(ϵᵀϵ)^(κ.d) + one(T))
end

arguments(κ::LogKernel) = (κ.d,)
isposdef_kernel(κ::LogKernel) = false

formula_string(κ::LogKernel) = "-‖x-y‖ᵈ"
argument_string(κ::LogKernel) = "d = $(κ.d)"
function description_string{T<:FloatingPoint}(κ::LogKernel{T}, eltype::Bool = true)
    "LogKernel" * (eltype ? "{$(T)}" : "") * "(d=$(κ.d))"
end

function description(κ::LogKernel)
    print(
        """ 
         Log Kernel:
         
         The power kernel is a positive semidefinite kernel. The function is
         given by:

             k(x,y) = -log(‖x-y‖ᵈ + 1)    x ∈ ℝⁿ, y ∈ ℝⁿ, d > 0
        """
    )
end


#==========================================================================
  Scalar Product Kernels
==========================================================================#

abstract ScalarProductKernel{T<:FloatingPoint} <: StandardKernel{T}

is_scalar_product(κ::ScalarProductKernel) = true

function kernel_function{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::Vector{T},
                                                   y::Vector{T})
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

kernelize_scalar{T<:FloatingPoint}(κ::LinearKernel, xᵀy::T) = xᵀy + κ.c

arguments(κ::LinearKernel) = (κ.c,)
isposdef_kernel(κ::LinearKernel) = true

formula_string(κ::LinearKernel) = "k(x,y) = xᵀy + c"
argument_string(κ::LinearKernel) = "c = $(κ.c)"
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
    PolynomialKernel{T}(α, c, convert(T, 2))
end

PolynomialKernel{T<:FloatingPoint}(α::T, c::T, d::Integer) = PolynomialKernel(α, c, convert(T, d))

function convert{T<:FloatingPoint}(::Type{PolynomialKernel{T}}, κ::PolynomialKernel)
    PolynomialKernel(convert(T, κ.α), convert(T, κ.c), convert(T, κ.d))
end

function kernelize_scalar{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T)
    (κ.α*xᵀy + κ.c)^κ.d
end

arguments(κ::PolynomialKernel) = (κ.α, κ.c, κ.d)
isposdef_kernel(κ::PolynomialKernel) = true

formula_string(κ::PolynomialKernel) = "(αxᵀy + c)ᵈ"
argument_string(κ::PolynomialKernel) = "α = $(κ.α), c = $(κ.c) and d = $(κ.d)"
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

kernelize_scalar{T<:FloatingPoint}(κ::SigmoidKernel, xᵀy::T) = tanh(κ.α*xᵀy + κ.c)

arguments(κ::SigmoidKernel) = (κ.α, κ.c)
isposdef_kernel(κ::SigmoidKernel) = false

formula_string(κ::SigmoidKernel) = "tanh(α‖x-y‖² + c)"
argument_string(κ::SigmoidKernel) = "α = $(κ.α) and c = $(κ.c)"
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

for (kernel, kernel_type) in ((:LinearKernel, :ScalarProductKernel),
                              (:PolynomialKernel, :ScalarProductKernel),
                              (:SigmoidKernel, :ScalarProductKernel),
                              (:GaussianKernel, :EuclideanDistanceKernel),
                              (:LaplacianKernel, :EuclideanDistanceKernel),
                              (:RationalQuadraticKernel, :EuclideanDistanceKernel),
                              (:MultiQuadraticKernel, :EuclideanDistanceKernel),
                              (:InverseMultiQuadraticKernel, :EuclideanDistanceKernel),
                              (:PowerKernel, :EuclideanDistanceKernel),
                              (:LogKernel, :EuclideanDistanceKernel))

    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, κ::$kernel)
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


#==========================================================================
  Pointwise Product Kernel
==========================================================================#

# Will look into pointwise product kernel in future

#=
immutable PointwiseProductKernel{T<:FloatingPoint} <: StandardKernel{T}
    k::Function
    c::T
    posdef::Bool
    function PointwiseProductKernel(k::Function, c::T = 0.0, posdef::Bool = false)
        method_exists(k, (Array{T},)) || error("k = $(k) must map f: ℝⁿ → ℝ (define method for" * (
                                               "Array{Float32} and Array{Float64}."))
        c >= 0 || error("c = $(c) must be non-negative.")
        new(k, c, posdef)
    end
end
function PointwiseProductKernel{T<:FloatingPoint}(k::Function, c::T = 0.0, posdef::Bool = false)
    PointwiseProductKernel{T}(k, c, posdef)
end

function convert{T<:FloatingPoint}(::Type{PointwiseProductKernel{T}}, κ::PointwiseProductKernel)
    PointwiseProductKernel(κ.k, convert(T, κ.c), κ.posdef)
end

@inline function kernel_function{T<:FloatingPoint}(K::PointwiseProductKernel{T}, x::Vector{T}, 
                                                   y::Vector{T})
    κ.k(x) * κ.k(y) + κ.c
end

arguments(κ::PointwiseProductKernel) = (κ.f,)
isposdef_kernel(κ::PointwiseProductKernel) = κ.posdef

formula_string(κ::PointwiseProductKernel) = "k(x,y) = f(x)f(y) + c"
argument_string(κ::PointwiseProductKernel) = "k, c=$(κ.c)"
description_string(κ::PointwiseProductKernel) = "PointwiseProductKernel(k, c=$(κ.c))"

function description(κ::PointwiseProductKernel)
    print(
        """ 
         Pointwise Product Kernel:
         
         The pointwise product kernel is the product of a real-valued multi-
         variate function applied to each of the vector arguments:

             k(x,y) = f(x)f(y) + c    x ∈ ℝⁿ, y ∈ ℝⁿ, c ∈ ℝ, f: ℝⁿ → ℝ
        """
    )
end

function convert{T<:FloatingPoint}(::Type{StandardKernel{T}}, κ::PointwiseProductKernel)
    PointwiseProductKernel(κ.k, convert(T, κ.c), κ.posdef)
end
function convert{T<:FloatingPoint}(::Type{SimpleKernel{T}}, κ::PointwiseProductKernel)
    PointwiseProductKernel(κ.k, convert(T, κ.c), κ.posdef)
end
=#
