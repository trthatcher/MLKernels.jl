#===================================================================================================
  Mercer Kernel Type & Related Functions
===================================================================================================#

abstract MercerKernel

abstract SimpleMercerKernel <: MercerKernel
abstract CompositeMercerKernel <: MercerKernel


abstract StandardMercerKernel <: SimpleMercerKernel
abstract TransformedMercerKernel <: SimpleMercerKernel

ScalableMercerKernel = Union(StandardMercerKernel, TransformedMercerKernel)

call{T<:FloatingPoint}(kernel::MercerKernel, x::Array{T}, y::Array{T}) = kernelfunction(kernel)(x, y)


#===================================================================================================
  Transformed and Scaled Mercer Kernels
===================================================================================================#

#== Exponential Mercer Kernel ====================#

type ExponentialMercerKernel <: TransformedMercerKernel
    κ::StandardMercerKernel
    ExponentialMercerKernel(κ::StandardMercerKernel) = new(κ)
end

kernelfunction(ψ::ExponentialMercerKernel) = exp(kernelfunction(ψ.κ)(x, y))

description_string(ψ::ExponentialMercerKernel) = "exp(" * description_string(ψ.κ) * ")"

function show(io::IO, ψ::ExponentialMercerKernel)
    println(io, "Exponential Mercer Kernel:")
    print(io, " " * description_string(ψ))
end

exp(κ::StandardMercerKernel) = ExponentialMercerKernel(deepcopy(κ))


#== Exponentiated Mercer Kernel ====================#

type ExponentiatedMercerKernel <: TransformedMercerKernel
    κ::StandardMercerKernel
    a::Integer
    function ExponentiatedMercerKernel(κ::StandardMercerKernel, a::Real)
        a > 0 || error("a = $(a) must be a non-negative number.")
        new(κ, a)
    end
end

kernelfunction(ψ::ExponentiatedMercerKernel) = (kernelfunction(ψ.κ)(x, y)) ^ ψ.a

description_string(ψ::ExponentiatedMercerKernel) = description_string(ψ.κ) * " ^ $(ψ.a)"

function show(io::IO, ψ::ExponentiatedMercerKernel)
    println(io, "Exponentiated Mercer Kernel:")
    print(io, " " * description_string(ψ))
end

^(κ::StandardMercerKernel, a::Integer) = ExponentiatedMercerKernel(deepcopy(κ), a)


#== Scaled Mercer Kernel ====================#

type ScaledMercerKernel <: SimpleMercerKernel
    a::Real
    κ::ScalableMercerKernel
    function ScaledMercerKernel(a::Real, κ::ScalableMercerKernel)
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ)
    end
end

function kernelfunction(ψ::ScaledMercerKernel)
    if ψ.a == 1
        k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernelfunction(ψ.κ)(x, y)
    else
        k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernelfunction(ψ.κ)(x, y) * convert(T, ψ.a)
    end
    return k
end

description_string(ψ::ScaledMercerKernel) = "$(ψ.a) * " * description_string(ψ.κ)

function show(io::IO, ψ::ScaledMercerKernel)
    println(io, "Scaled Mercer Kernel:")
    print(io, " " * description_string(ψ))
end

*(a::Real, κ::ScalableMercerKernel) = ScaledMercerKernel(a, deepcopy(κ))
*(κ::ScalableMercerKernel, a::Real) = *(a, κ)

*(a::Real, ψ::ScaledMercerKernel) = ScaledMercerKernel(a * ψ.a, deepcopy(ψ.κ))
*(ψ::ScaledMercerKernel, a::Real) = *(a, ψ)


#===================================================================================================
  Composite Mercer Kernels
===================================================================================================#

#== Mercer Kernel Product ====================#

type MercerKernelProduct <: CompositeMercerKernel
    a::Real
    ψ₁::ScalableMercerKernel
    ψ₂::ScalableMercerKernel
    function MercerKernelProduct(a::Real, ψ₁::ScalableMercerKernel, ψ₂::ScalableMercerKernel)
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, ψ₁, ψ₂)
    end
end

function description_string(ψ::MercerKernelProduct) 
    if ψ.a == 1
        return description_string(ψ.ψ₁) * " * " * description_string(ψ.ψ₂)
    else
        return "$(ψ.a) * " * description_string(ψ.ψ₁) * " * " * description_string(ψ.ψ₂)
    end
end

function show(io::IO, ψ::MercerKernelProduct)
    println(io, "Mercer Kernel Product:")
    print(io, " " * description_string(ψ))
end

function kernelfunction(ψ::MercerKernelProduct)
    if ψ.a == 1
        k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernelfunction(ψ.κ₁)(x, y) * kernelfunction(ψ.κ₂)(x, y)
    else
        k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernelfunction(ψ.κ₁)(x, y) * kernelfunction(ψ.κ₂)(x, y) * convert(T, ψ.a)
    end
    return k
end

*(κ₁::ScalableMercerKernel, κ₂::ScalableMercerKernel) = MercerKernelProduct(1, deepcopy(κ₁), deepcopy(κ₂))
*(κ::ScalableMercerKernel, ψ::ScaledMercerKernel) = MercerKernelProduct(ψ.a, deepcopy(κ), deepcopy(ψ.κ))
*(ψ::ScaledMercerKernel, κ::ScalableMercerKernel) = MercerKernelProduct(ψ.a, deepcopy(ψ.κ), deepcopy(κ))
*(ψ₁::ScaledMercerKernel, ψ₂::ScaledMercerKernel) = MercerKernelProduct(ψ₁.a * ψ₂.a, deepcopy(ψ₁.κ), deepcopy(ψ₂.κ))

*(a::Real, ψ::MercerKernelProduct) = MercerKernelProduct(a * ψ.a, deepcopy(ψ.ψ₁), deepcopy(ψ.ψ₂))
*(ψ::MercerKernelProduct, a::Real) = a * ψ


#== Mercer Kernel Sum ====================#

type MercerKernelSum <: CompositeMercerKernel
    ψ₁::SimpleMercerKernel
    ψ₂::SimpleMercerKernel
    function MercerKernelSum(ψ₁::SimpleMercerKernel, ψ₂::SimpleMercerKernel)
        new(ψ₁,ψ₂)
    end
end

function kernelfunction(ψ::MercerKernelSum) 
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernelfunction(ψ.ψ₁)(x, y) + kernelfunction(ψ.ψ₂)(x, y)
    return k
end

function show(io::IO, ψ::MercerKernelSum)
    println(io, "Mercer Kernel Sum:")
    print(io, " " * description_string(ψ.ψ₁) * " + " * description_string(ψ.ψ₂))
end

+(ψ₁::SimpleMercerKernel, ψ₂::SimpleMercerKernel)  = MercerKernelSum(deepcopy(ψ₁), deepcopy(ψ₂))

*(a::Real, ψ::MercerKernelSum) = (a * ψ.ψ₁) + (a * ψ.ψ₂)
*(ψ::MercerKernelSum, a::Real) = a * ψ


#===================================================================================================
  Standard Mercer Kernels
===================================================================================================#

function show(io::IO, obj::StandardMercerKernel)
    print(io, " " * description_string(obj))
end

#== Pointwise Product Kernel ====================#

function pointwiseproductkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, f::Function)
    return f(x) * f(y)
end

type PointwiseProductKernel <: StandardMercerKernel
    f::Function
    function PointwiseProductKernel(f::Function)
        method_exists(f, (Array{Float32},)) && method_exists(f, (Array{Float64},)) || error("f = $(f) must map f: ℝⁿ → ℝ (define methods for both Array{Float32} and Array{Float64}).")
        new(f)
    end
end

arguments(obj::PointwiseProductKernel) = obj.f
function kernelfunction(obj::PointwiseProductKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = pointwiseproductkernel(x, y, copy(obj.f))
    return k
end

formula_string(obj::PointwiseProductKernel) = "k(x,y) = f(x)f(y)"
argument_string(obj::PointwiseProductKernel) = "f = $(obj.f)"
description_string(obj::PointwiseProductKernel) = "PointwiseProductKernel(f=$(obj.f))"

function description(obj::PointwiseProductKernel)
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

type GenericKernel <: StandardMercerKernel
    k::Function
    function GenericKernel(k::Function)
        method_exists(f, (Array{Float32}, Array{Float32})) && method_exists(f, (Array{Float64}, Array{Float64})) || error("k = $(f) must map k: ℝⁿ×ℝⁿ → ℝ (define methods for both Array{Float32} and Array{Float64}).")
        new(k)
    end
end

arguments(obj::GenericKernel) = obj.k
function kernelfunction(obj::GenericKernel)
    return copy(obj.k)
end

formula_string(obj::GenericKernel) = "k(x,y)"
argument_string(obj::GenericKernel) = "k = $(obj.k)"
description_string(obj::GenericKernel) = "GenericKernel(k=$(obj.k))"

function description(obj::GenericKernel)
    print(
        """ 
         Generic Kernel:
         ===================================================================
         Customized definition:

             k: ℝⁿ×ℝⁿ → ℝ
        """
    )
end


#== Linear Kernel ====================#

function linearkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    return BLAS.dot(length(x), x, 1, y, 1)
end

function linearkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, c::Real)
    return BLAS.dot(length(x), x, 1, y, 1) + convert(T,c)
end

type LinearKernel <: StandardMercerKernel
    c::Real
    function LinearKernel(c::Real=0)
        c >= 0 || error("c = $c must be greater than zero.")
        new(c)
    end
end

arguments(obj::LinearKernel) = obj.c
function kernelfunction(obj::LinearKernel)
    if obj.c == 0
        k(x,y) = linearkernel(x, y)
    else
        k(x,y) = linearkernel(x, y, obj.c)
    end
    return k
end

formula_string(obj::LinearKernel) = "k(x,y) = xᵗy + c"
argument_string(obj::LinearKernel) = "c = $(obj.c)"
description_string(obj::LinearKernel) = "LinearKernel(c=$(obj.c))"

function description(obj::LinearKernel)
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

function generalizedpolynomialkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, α::Real, c::Real, d::Real)
    return (convert(T,α)*BLAS.dot(length(x), x, 1, y, 1) + convert(T,c))^convert(T,d)
end

function polynomialkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, c::Real, d::Real)
    return (BLAS.dot(length(x), x, 1, y, 1) + convert(T, c))^convert(T, d)
end

function homogenouspolynomialkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, d::Real)
    return (BLAS.dot(length(x), x, 1, y, 1))^convert(T, d)
end

type PolynomialKernel <: StandardMercerKernel
    α::Real
    c::Real
    d::Real
    function PolynomialKernel(α::Real=1,c::Real=1,d::Real=2)
        α > 0 || error("α = $(α) must be greater than zero.")
        c >= 0 || error("c = $(c) must be a non-negative number.")
        d >= 0 || error("d = $(d) must be a non-negative number.") 
        new(α, c, d)
    end
end

arguments(obj::PolynomialKernel) = (obj.α, obj.c, obj.d)
function kernelfunction(obj::PolynomialKernel)
    if obj.α == 1
        if obj.c == 0
            k(x,y) = homogeneouspolynomialkernel(x, y, obj.d)
        else
            k(x,y) = polynomialkernel(x, y, obj.c, obj.d)
        end
    else
        k(x,y) = generalizedpolynomialkernel(x, y, obj.α, obj.c, obj.d)
    end
    return k
end

formula_string(obj::PolynomialKernel) = "(αxᵗy + c)ᵈ"
argument_string(obj::PolynomialKernel) = "α = $(obj.α), c = $(obj.c) and d = $(obj.d)"
description_string(obj::PolynomialKernel) = "PolynomialKernel(α=$(obj.α),c=$(obj.c),d=$(obj.d))"

function description(obj::PolynomialKernel)
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


#== Gaussian Kernel ===============#

function gaussiankernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, η::Real)
    δ = x .- y
    return exp(- convert(T, η) * BLAS.dot(length(x), δ, 1, δ, 1))
end

type GaussianKernel <: StandardMercerKernel
    η::Real
    function GaussianKernel(η::Real=1)
        η > 0 || error("σ = $(η) must be greater than 0.")
        new(η)
    end
end

arguments(obj::GaussianKernel) = obj.η
function kernelfunction(obj::GaussianKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = gaussiankernel(x, y, obj.η)
    return k
end

formula_string(obj::GaussianKernel) = "exp(-η‖x-y‖²)"
argument_string(obj::GaussianKernel) = "η = $(obj.η)"
description_string(obj::GaussianKernel) = "GaussianKernel(η=$(obj.η))"

function description(obj::GaussianKernel)
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
    return exp(- convert(T, η) * BLAS.nrm2(n, ϵ, 1))
end

type LaplacianKernel <: StandardMercerKernel
    η::Real
    function LaplacianKernel(η::Real=1)
        η > 0 || error("η = $(η) must be greater than zero.")
        new(η)
    end
end

arguments(obj::LaplacianKernel) = obj.η
function kernelfunction(obj::LaplacianKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = laplaciankernel(x, y, obj.η)
    return k
end

formula_string(obj::LaplacianKernel) = "exp(-η‖x-y‖)"
argument_string(obj::LaplacianKernel) = "η = $(obj.η)"
description_string(obj::LaplacianKernel) = "LaplacianKernel(η=$(obj.η))"

function description(obj::LaplacianKernel)
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


#== Sigmoid Kernel ===============#

function sigmoidkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, α::Real, c::Real)
    return tanh(convert(T,α) * BLAS.dot(length(x), x, 1, y, 1) + convert(T,c))
end

type SigmoidKernel <: StandardMercerKernel
    α::Real
    c::Real
    function SigmoidKernel(α::Real=1, c::Real=0)
        α > 0 || error("α = $(α) must be greater than zero.")
        c >= 0 || error("c = $(c) must be non-negative.")
        new(α, c)
    end
end

arguments(obj::SigmoidKernel) = (obj.α, obj.c)
function kernelfunction(obj::SigmoidKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = sigmoidkernel(x, y, obj.α, obj.c)
    return k
end

formula_string(obj::SigmoidKernel) = "tanh(α‖x-y‖² + c)"
argument_string(obj::SigmoidKernel) = "α = $(obj.α) and c = $(obj.c)"
description_string(obj::SigmoidKernel) = "SigmoidKernel(α=$(obj.α),c=$(obj.c))"

function description(obj::SigmoidKernel)
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


#== Rational Quadratic Kernel ===============#

function rationalquadratickernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, c::Real)
    n = length(x)
    ϵ = BLAS.axpy!(n, convert(T, -1), y, 1, copy(x), 1)
    d² = BLAS.dot(n, ϵ, 1, ϵ, 1)
    return convert(T, 1) - d²/(d² + convert(T, c))
end

type RationalQuadraticKernel <: StandardMercerKernel
    c::Real
    function RationalQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

arguments(obj::RationalQuadraticKernel) = obj.c
function kernelfunction(obj::RationalQuadraticKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = rationalquadratickernel(x, y, obj.c)
    return k
end

formula_string(obj::RationalQuadraticKernel) = "1 - ‖x-y‖²/(‖x-y‖² + c)"
argument_string(obj::RationalQuadraticKernel) = "c = $(obj.c)"
description_string(obj::RationalQuadraticKernel) = "RationalQuadraticKernel(c=$(obj.c))"

function description(obj::RationalQuadraticKernel)
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

type MultiQuadraticKernel <: StandardMercerKernel
    c::Real
    function MultiQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

arguments(obj::MultiQuadraticKernel) = obj.c
function kernelfunction(obj::MultiQuadraticKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = multiquadratickernel(x, y, obj.c)
    return k
end

formula_string(obj::MultiQuadraticKernel) = "√(‖x-y‖² + c)"
argument_string(obj::MultiQuadraticKernel) = "c = $(obj.c)"
description_string(obj::MultiQuadraticKernel) = "MultiQuadraticKernel(c=$(obj.c))"

function description(obj::MultiQuadraticKernel)
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

type InverseMultiQuadraticKernel <: StandardMercerKernel
    c::Real
    function InverseMultiQuadraticKernel(c::Real=1)
        c > 0 || error("c = $(c) must be greater than zero.")
        new(c)
    end
end

arguments(obj::InverseMultiQuadraticKernel) = obj.c
function kernelfunction(obj::InverseMultiQuadraticKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = inversemultiquadratickernel(x, y, obj.c)
    return k
end

formula_string(obj::InverseMultiQuadraticKernel) = "1/√(‖x-y‖² + c)"
argument_string(obj::InverseMultiQuadraticKernel) = "c = $(obj.c)"
description_string(obj::InverseMultiQuadraticKernel) = "InverseMultiQuadraticKernel(c=$(obj.c))"

function description(obj::InverseMultiQuadraticKernel)
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
    return -BLAS.dot(length(x), ϵ, 1, ϵ, 1)^convert(T,d)
end

type PowerKernel <: StandardMercerKernel
    d::Real
    function PowerKernel(d::Real = 1)
        d > 0 || error("d = $(d) must be greater than zero.")
        new(d)
    end
end

arguments(obj::PowerKernel) = obj.d
function kernelfunction(obj::PowerKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = powerkernel(x, y, obj.d)
    return k
end

formula_string(obj::PowerKernel) = "-‖x-y‖ᵈ"
argument_string(obj::PowerKernel) = "d = $(obj.d)"
description_string(obj::PowerKernel) = "PowerKernel(d=$(obj.d))"

function description(obj::PowerKernel)
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







function logkernel{T<:FloatingPoint}(x::Array{T},y::Array{T},d::Real)
    δ = x .- y
    return - log(BLAS.dot(length(x), δ, 1, δ, 1)^convert(T,d) + 1)
end
type LogKernel <: MercerKernel
    k::Function
    d::Real
    function LogKernel(d::Real=1)
        k(x,y) = logkernel(x,y,d)
        new(k,d)
    end
end
function show(io::IO,obj::LogKernel)
    print("Power Kernel: k(x,y) = -log(‖x-y‖ᵈ + 1)   with d = $(obj.d)")
end


function splinekernel{T<:FloatingPoint}(x::Array{T},y::Array{T})
    xy = x .* y
    min_xy = min(x,y)
    v = 1 .+ xy .* (1 .+ min_xy) .- (x + y)/2 .* min_xy.^2 + min_xy .^ 3
    return prod(v)
end
type SplineKernel <: MercerKernel
    k::Function
    function SplineKernel()
        k(x,y) = splinekernel(x,y)
        new(k)
    end
end
function show(io::IO,obj::LogKernel)
    print("Spline Kernel: see reference for k(x,y)")
end
