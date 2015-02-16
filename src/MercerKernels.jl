#===================================================================================================
  Mercer Kernel Type & Related Functions
===================================================================================================#

abstract MercerKernel

abstract StandardMercerKernel <: MercerKernel
abstract CompositeMercerKernel <: MercerKernel


#== Product Kernel ====================#

function productkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, kernels::Array{StandardMercerKernel}, a::Real = 1)
    value = convert(T, a)
    for kernel in kernels
        value *= kernelfunction(kernel)(x, y)
    end
    return value
end

type ProductMercerKernel <: CompositeMercerKernel
    a::Real
    kernels::Array{StandardMercerKernel}
    function ProductMercerKernel(a::Real, kernels::Array{StandardMercerKernel})
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, kernels)
    end
end
ProductMercerKernel(a::Real, kernels::StandardMercerKernel...) = ProductMercerKernel(a, StandardMercerKernel[deepcopy(kernels)...])
ProductMercerKernel(kernels::StandardMercerKernel...) = ProductMercerKernel(1, kernels...)

function kernelfunction(obj::ProductMercerKernel)
    if length(obj.kernels) == 1
        if obj.a == 0
            k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernelfunction(obj.kernels[1])(x, y)
        else
            k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = convert(T, obj.a) * kernelfunction(obj.kernels[1])(x, y)
        end
    elseif length(obj.kernels) == 2
        if obj.a == 0
            k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernelfunction(obj.kernels[1])(x, y) * kernelfunction(obj.kernels[2])(x, y)
        else
            k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = convert(T, obj.a) * (kernelfunction(obj.kernels[1])(x, y) * kernelfunction(obj.kernels[2])(x, y))
        end
    else
        k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = productkernel(x, y, deepcopy(obj.kernels), obj.a)
    end
    return k
end

function description_string(obj::ProductMercerKernel)
    n = length(obj.kernels)
    if n == 1
        if obj.a == 1
            description = description_string(obj.kernels[1])
        else
            description = description_string(obj.kernels[1]) * " * $(obj.a)"
        end
    else
        description = "Product(" * description_string(obj.kernels[1])
        if n > 1
            if n >= 3
                description *= ", ..."
            end
            description *= ", " * description_string(obj.kernels[n])
        end
        if obj.a == 1
            description *= ")"
        else
            description *= ") * $(obj.a)"
        end
    end
    return description
end

function show(io::IO, obj::ProductMercerKernel)
    println(io, "$(length(obj.kernels))-element Mercer Kernel Product:")
    n = length(obj.kernels)
    if obj.a == 1
        println(io, " " * description_string(obj.kernels[1]))
    else
        println(io," $(obj.a)") 
        println(" * " * description_string(obj.kernels[1]))
    end
    if n >= 2
        for i = 2:(n-1)
            println(io, " * " * description_string(obj.kernels[i]))
        end
        print(io, " * " * description_string(obj.kernels[n]))
    end
end

*(a::Real, kernel::StandardMercerKernel) = ProductMercerKernel(a, kernel)
*(kernel::StandardMercerKernel, a::Real) = a * kernel
*(a::Real, kernel::ProductMercerKernel) = ProductMercerKernel(a * kernel.a, deepcopy(kernel.kernels))
*(kernel::ProductMercerKernel, a::Real) = a * kernel

*(lkernel::StandardMercerKernel, rkernel::StandardMercerKernel) = ProductMercerKernel(lkernel, rkernel)
*(lkernel::StandardMercerKernel, rkernel::ProductMercerKernel) = ProductMercerKernel(rkernel.a, lkernel, rkernel.kernels...)
*(lkernel::ProductMercerKernel, rkernel::StandardMercerKernel) = ProductMercerKernel(lkernel.a, lkernel.kernels..., rkernel)
*(lkernel::ProductMercerKernel, rkernel::ProductMercerKernel) = ProductMercerKernel(lkernel.a * rkernel.a, lkernel.kernels..., rkernel.kernels...)

#== Sum Kernel ====================#

function sumkernel{T<:FloatingPoint}(x::Array{T}, y::Array{T}, kernels::Array{ProductMercerKernel})
    value = convert(T, 0)
    for kernel in kernels
        value += kernelfunction(kernel)(x, y)
    end
    return value
end

type SumMercerKernel <: CompositeMercerKernel
    kernels::Array{ProductMercerKernel}
end

function kernelfunction(obj::SumMercerKernel)
    if length(obj.kernels) == 1
        k = kernelfunction(obj.kernels[1])
    elseif length(obj.kernels) == 2
        k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = kernelfunction(obj.kernels[1])(x, y) + kernelfunction(obj.kernels[2])(x, y)
    else
        k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = sumkernel(x, y, deepcopy(obj.kernels))
    end
    return k
end

function show(io::IO, obj::SumMercerKernel)
	println(io, "$(length(obj.kernels))-element Mercer Kernel Summation:")
    n = length(obj.kernels)
    println(io, " " * description_string(obj.kernels[1]))
    if n >= 2
        for i = 2:(n-1)
            println(io, " + " * description_string(obj.kernels[i]))
        end
        print(io, " + " * description_string(obj.kernels[n]))
    end
end

+(lkernel::StandardMercerKernel, rkernel::StandardMercerKernel) = SumMercerKernel([ProductMercerKernel(lkernel), ProductMercerKernel(rkernel)])
+(lkernel::ProductMercerKernel, rkernel::StandardMercerKernel) = SumMercerKernel([deepcopy(lkernel), ProductMercerKernel(rkernel)])
+(lkernel::StandardMercerKernel, rkernel::ProductMercerKernel) = SumMercerKernel([ProductMercerKernel(lkernel), deepcopy(rkernel)])
+(lkernel::ProductMercerKernel, rkernel::ProductMercerKernel) = SumMercerKernel([deepcopy(lkernel), deepcopy(rkernel)])

+(lkernel::SumMercerKernel, rkernel::StandardMercerKernel) = SumMercerKernel([deepcopy(lkernel.kernels)..., ProductMercerKernel(rkernel)])
+(lkernel::SumMercerKernel, rkernel::ProductMercerKernel) = SumMercerKernel([deepcopy(lkernel.kernels)..., deepcopy(rkernel)])
+(lkernel::SumMercerKernel, rkernel::SumMercerKernel) = SumMercerKernel([deepcopy(lkernel.kernels)..., deepcopy(rkernel.kernels)...])

+(lkernel::StandardMercerKernel, rkernel::SumMercerKernel) = SumMercerKernel([ProductMercerKernel(lkernel), deepcopy(rkernel.kernels)...])
+(lkernel::ProductMercerKernel, rkernel::SumMercerKernel) = SumMercerKernel([deepcopy(lkernel), deepcopy(rkernel.kernels)...])


#===================================================================================================
  Standard Mercer Kernels
===================================================================================================#

function show(io::IO, obj::StandardMercerKernel)
	print(io, description_string(obj))
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

###########################################3


function powerkernel{T<:FloatingPoint}(x::Array{T},y::Array{T},d::Real)
	δ = x .- y
	return -BLAS.dot(length(x), δ, 1, δ, 1)^convert(T,d)
end
type PowerKernel <: MercerKernel
	k::Function
	d::Real
	function PowerKernel(d::Real=1)
		k(x,y) = powerkernel(x,y,d)
		new(k,d)
	end
end
function show(io::IO,obj::PowerKernel)
	print("Power Kernel: k(x,y) = -‖x-y‖ᵈ   with d = $(obj.d)")
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
