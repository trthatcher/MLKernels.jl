#===================================================================================================
  Mercer Kernel Type & Related Functions
===================================================================================================#

abstract Kernel

abstract SimpleKernel <: Kernel
abstract CompositeKernel <: Kernel

abstract StandardKernel <: SimpleKernel
abstract TransformedKernel <: SimpleKernel

ScalableKernel = Union(StandardKernel, TransformedKernel)

call{T<:FloatingPoint}(κ::Kernel, x::Array{T}, y::Array{T}) = kernelfunction(κ)(x, y)


#===================================================================================================
  Transformed and Scaled Mercer Kernels
===================================================================================================#

#== Exponential Mercer Kernel ====================#

type ExponentialKernel <: TransformedKernel
    κ::StandardKernel
    ExponentialKernel(κ::StandardKernel) = new(κ)
end

@inline kernelfunction(ψ::ExponentialKernel) = exp(kernelfunction(ψ.κ)(x, y))

description_string(ψ::ExponentialKernel) = "exp(" * description_string(ψ.κ) * ")"

function show(io::IO, ψ::ExponentialKernel)
    println(io, "Exponential Mercer Kernel:")
    print(io, " " * description_string(ψ))
end

exp(κ::StandardKernel) = ExponentialKernel(deepcopy(κ))


#== Exponentiated Mercer Kernel ====================#

type ExponentiatedKernel <: TransformedKernel
    κ::StandardKernel
    a::Integer
    function ExponentiatedKernel(κ::StandardKernel, a::Real)
        a > 0 || error("a = $(a) must be a non-negative number.")
        new(κ, a)
    end
end

@inline kernelfunction(ψ::ExponentiatedKernel) = (kernelfunction(ψ.κ)(x, y)) ^ ψ.a

description_string(ψ::ExponentiatedKernel) = description_string(ψ.κ) * " ^ $(ψ.a)"

function show(io::IO, ψ::ExponentiatedKernel)
    println(io, "Exponentiated Mercer Kernel:")
    print(io, " " * description_string(ψ))
end

^(κ::StandardKernel, a::Integer) = ExponentiatedKernel(deepcopy(κ), a)


#== Scaled Mercer Kernel ====================#

type ScaledKernel <: SimpleKernel
    a::Real
    κ::ScalableKernel
    function ScaledKernel(a::Real, κ::ScalableKernel)
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ)
    end
end

@inline function kernelfunction(ψ::ScaledKernel)
    if ψ.a == 1
        k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernelfunction(ψ.κ)(x, y)
    end
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernelfunction(ψ.κ)(x, y) * convert(T, ψ.a)
end

description_string(ψ::ScaledKernel) = "$(ψ.a) * " * description_string(ψ.κ)

function show(io::IO, ψ::ScaledKernel)
    println(io, "Scaled Mercer Kernel:")
    print(io, " " * description_string(ψ))
end

*(a::Real, κ::ScalableKernel) = ScaledKernel(a, deepcopy(κ))
*(κ::ScalableKernel, a::Real) = *(a, κ)

*(a::Real, ψ::ScaledKernel) = ScaledKernel(a * ψ.a, deepcopy(ψ.κ))
*(ψ::ScaledKernel, a::Real) = *(a, ψ)


#===================================================================================================
  Composite Kernels
===================================================================================================#

#== Mercer Kernel Product ====================#

type KernelProduct <: CompositeKernel
    a::Real
    ψ₁::ScalableKernel
    ψ₂::ScalableKernel
    function KernelProduct(a::Real, ψ₁::ScalableKernel, ψ₂::ScalableKernel)
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, ψ₁, ψ₂)
    end
end

function description_string(ψ::KernelProduct) 
    if ψ.a == 1
        return description_string(ψ.ψ₁) * " * " * description_string(ψ.ψ₂)
    end
    "$(ψ.a) * " * description_string(ψ.ψ₁) * " * " * description_string(ψ.ψ₂)
end

function show(io::IO, ψ::KernelProduct)
    println(io, "Mercer Kernel Product:")
    print(io, " " * description_string(ψ))
end

@inline function kernelfunction(ψ::KernelProduct)
    if ψ.a == 1
        k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = (
            kernelfunction(ψ.κ₁)(x, y) * kernelfunction(ψ.κ₂)(x, y))
    end
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = (
        kernelfunction(ψ.κ₁)(x, y) * kernelfunction(ψ.κ₂)(x, y) * convert(T, ψ.a))
end

*(κ₁::ScalableKernel, κ₂::ScalableKernel) = (
    KernelProduct(1, deepcopy(κ₁), deepcopy(κ₂)))

*(κ::ScalableKernel, ψ::ScaledKernel) = (
    KernelProduct(ψ.a, deepcopy(κ), deepcopy(ψ.κ)))

*(ψ::ScaledKernel, κ::ScalableKernel) = (
    KernelProduct(ψ.a, deepcopy(ψ.κ), deepcopy(κ)))

*(ψ₁::ScaledKernel, ψ₂::ScaledKernel) = (
    KernelProduct(ψ₁.a * ψ₂.a, deepcopy(ψ₁.κ), deepcopy(ψ₂.κ)))

*(a::Real, ψ::KernelProduct) = (
    KernelProduct(a * ψ.a, deepcopy(ψ.ψ₁), deepcopy(ψ.ψ₂)))

*(ψ::KernelProduct, a::Real) = a * ψ


#== Mercer Kernel Sum ====================#

type KernelSum <: CompositeKernel
    ψ₁::SimpleKernel
    ψ₂::SimpleKernel
    KernelSum(ψ₁::SimpleKernel, ψ₂::SimpleKernel) = new(ψ₁,ψ₂)
end

@inline function kernelfunction(ψ::KernelSum)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = (
        kernelfunction(ψ.ψ₁)(x, y) + kernelfunction(ψ.ψ₂)(x, y))
end

function show(io::IO, ψ::KernelSum)
    println(io, "Mercer Kernel Sum:")
    print(io, " " * description_string(ψ.ψ₁) * " + " * description_string(ψ.ψ₂))
end

+(ψ₁::SimpleKernel, ψ₂::SimpleKernel) = KernelSum(deepcopy(ψ₁), deepcopy(ψ₂))

*(a::Real, ψ::KernelSum) = (a * ψ.ψ₁) + (a * ψ.ψ₂)
*(ψ::KernelSum, a::Real) = a * ψ


#===================================================================================================
  Standard Kernels
===================================================================================================#

function show(io::IO, κ::StandardKernel)
    print(io, " " * description_string(κ))
end


#=================================================
  Stationary Kernels
=================================================#

abstract StationaryKernel <: StandardKernel


#== Gaussian Kernel ===============#

function gaussiankernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, η::Real)
    δ = x .- y
    return exp(- convert(T, η) * BLAS.dot(length(x), δ, 1, δ, 1))
end

type GaussianKernel <: StationaryKernel
    η::Real
    function GaussianKernel(η::Real=1)
        η > 0 || error("σ = $(η) must be greater than 0.")
        new(η)
    end
end

arguments(κ::GaussianKernel) = κ.η

function kernelfunction(κ::GaussianKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = gaussiankernel(x, y, κ.η)
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

function laplaciankernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, η::Real)
    n = length(x)
    ϵ = BLAS.axpy!(n, convert(T, -1), y, 1, copy(x), 1)
    exp(- convert(T, η) * BLAS.nrm2(n, ϵ, 1))
end

type LaplacianKernel <: StationaryKernel
    η::Real
    function LaplacianKernel(η::Real=1)
        η > 0 || error("η = $(η) must be greater than zero.")
        new(η)
    end
end

arguments(κ::LaplacianKernel) = κ.η

@inline function kernelfunction(κ::LaplacianKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = laplaciankernel(x, y, κ.η)
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

function rationalquadratickernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, c::Real)
    n = length(x)
    ϵ = BLAS.axpy!(n, convert(T, -1), y, 1, copy(x), 1)
    d² = BLAS.dot(n, ϵ, 1, ϵ, 1)
    return convert(T, 1) - d²/(d² + convert(T, c))
end

type RationalQuadraticKernel <: StationaryKernel
    c::Real
    function RationalQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

arguments(κ::RationalQuadraticKernel) = κ.c

@inline function kernelfunction(κ::RationalQuadraticKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = rationalquadratickernel(x, y, κ.c)
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

function multiquadratickernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, c::Real)
    n = length(x)
    ϵ = BLAS.axpy!(n, convert(T, -1), y, 1, copy(x), 1)
    return sqrt(BLAS.dot(length(x), ϵ, 1, ϵ, 1) + convert(T,c))
end

type MultiQuadraticKernel <: StationaryKernel
    c::Real
    function MultiQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

arguments(κ::MultiQuadraticKernel) = κ.c

@inline function kernelfunction(κ::MultiQuadraticKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = multiquadratickernel(x, y, κ.c)
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

function inversemultiquadratickernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, c::Real)
    n = length(x)
    ϵ = BLAS.axpy!(n, convert(T, -1), y, 1, copy(x), 1)
    return 1 / (sqrt(BLAS.dot(n, ϵ, 1, ϵ, 1) + convert(T, c)))
end

type InverseMultiQuadraticKernel <: StandardKernel
    c::Real
    function InverseMultiQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

arguments(κ::InverseMultiQuadraticKernel) = κ.c

@inline function kernelfunction(κ::InverseMultiQuadraticKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = inversemultiquadratickernel(x, y, κ.c)
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

function powerkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, d::Real)
    n = length(x)
    ϵ = BLAS.axpy!(n, convert(T, -1), y, 1, copy(x), 1)
    return -BLAS.dot(n, ϵ, 1, ϵ, 1)^convert(T,d)
end

type PowerKernel <: StationaryKernel
    d::Real
    function PowerKernel(d::Real = 1)
        d > 0 || error("d = $(d) must be greater than zero.")
        new(d)
    end
end

arguments(κ::PowerKernel) = κ.d

@inline function kernelfunction(κ::PowerKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = powerkernel(x, y, κ.d)
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

function logkernel{T<:FloatingPoint}(x::Array{T},y::Array{T},d::Real)
    n = length(x)
    ϵ = BLAS.axpy!(n, convert(T, -1), y, 1, copy(x), 1)
    -log(BLAS.dot(n, ϵ, 1, ϵ, 1)^convert(T,d) + convert(T, 1))
end

type LogKernel <: StationaryKernel
    d::Real
    function LogKernel(d::Real = 1)
        d > 0 || error("d = $(d) must be greater than zero.")
        new(d)
    end
end

arguments(κ::LogKernel) = κ.d

@inline function kernelfunction(κ::LogKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = logkernel(x, y, κ.d)
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


#== Linear Kernel ====================#

@inline function linearkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, c::Real)
    BLAS.dot(length(x), x, 1, y, 1) + convert(T,c)
end

@inline linearkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = BLAS.dot(length(x), x, 1, y, 1)

type LinearKernel <: NonStationaryKernel
    c::Real
    function LinearKernel(c::Real=0)
        c >= 0 || error("c = $c must be greater than zero.")
        new(c)
    end
end

arguments(κ::LinearKernel) = κ.c

function kernelfunction(κ::LinearKernel)
    if κ.c == 0
        k(x,y) = linearkernel(x, y)
    end
    k(x,y) = linearkernel(x, y, κ.c)
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

@inline function polynomialkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, α::Real,
                                                    c::Real, d::Real)
    (convert(T,α)*BLAS.dot(length(x), x, 1, y, 1) + convert(T,c))^convert(T,d)
end

@inline function polynomialkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, c::Real, d::Real) 
    (BLAS.dot(length(x), x, 1, y, 1) + convert(T, c))^convert(T, d)
end

@inline function polynomialkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, d::Real)
    (BLAS.dot(length(x), x, 1, y, 1))^convert(T, d)
end

type PolynomialKernel <: NonStationaryKernel
    α::Real
    c::Real
    d::Real
    function PolynomialKernel(α::Real=1,c::Real=1,d::Real=2)
        α > 0 || error("α = $(α) must be greater than zero.")
        c >= 0 || error("c = $(c) must be a non-negative number.")
        d > 0 || error("d = $(d) must be greater than zero.") 
        new(α, c, d)
    end
end

arguments(κ::PolynomialKernel) = (κ.α, κ.c, κ.d)

function kernelfunction(κ::PolynomialKernel)
    if κ.α == 1
        if κ.c == 0
            return k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = polynomialkernel(x, y, κ.d)
        end
        return k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = polynomialkernel(x, y, κ.c, κ.d)
    end
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = polynomialkernel(x, y, κ.α, κ.c, κ.d)
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

@inline function sigmoidkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, α::Real, c::Real)
    tanh(convert(T,α) * BLAS.dot(length(x), x, 1, y, 1) + convert(T,c))
end

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

@inline function kernelfunction(κ::SigmoidKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = sigmoidkernel(x, y, κ.α, κ.c)
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

             k(x,y) = tanh(α‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0
        """
    )
end


#=================================================
  Generic Kernels
=================================================#

#== Pointwise Product Kernel ==#

@inline function pointwiseproductkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, f::Function)
    f(x) * f(y)
end

type PointwiseProductKernel <: StandardKernel
    f::Function
    function PointwiseProductKernel(f::Function)
        method_exists(f, (Array{Float32},)) && method_exists(f, (Array{Float64},)) || (
            error("f = $(f) must map f: ℝⁿ → ℝ (define methods for both Array{Float32} and " * ( 
                  "Array{Float64}).")))
        new(f)
    end
end

arguments(κ::PointwiseProductKernel) = κ.f

@inline function kernelfunction(κ::PointwiseProductKernel)
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


#== Generic Kernel ====================#

type GenericKernel <: StandardKernel
    k::Function
    function GenericKernel(k::Function)
        method_exists(f, (Array{Float32}, Array{Float32})) && (
            method_exists(f, (Array{Float64}, Array{Float64})) || (
            error("k = $(f) must map k: ℝⁿ×ℝⁿ → ℝ (define methods for both" * (
                  "Array{Float32} and Array{Float64})."))))
        new(k)
    end
end

arguments(κ::GenericKernel) = κ.k

kernelfunction(κ::GenericKernel) = copy(κ.k)

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
