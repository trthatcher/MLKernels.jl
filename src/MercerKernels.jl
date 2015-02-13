#===================================================================================================
  Mercer Kernel Type & Related Functions
===================================================================================================#

abstract MercerKernel

abstract StandardMercerKernel <: MercerKernel


#== Scaled Kernel ====================#

type ScaledMercerKernel <: MercerKernel
    a::Real
    kernel::StandardMercerKernel
    function ScaledMercerKernel(kernel::StandardMercerKernel, a::Real = 1)
        a > 0 || error("a = $a must be greater than zero.")
        new(a, kernel)
    end
end

function kernel(obj::ScaledMercerKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = convert(T, obj.a) * kernel(obj.kernel)(x, y)
    return k
end

function show(io::IO,obj::ScaledMercerKernel)
	println("Scaled Kernel: k(x,y) = ak₁(x,y) with a = $(obj.a)")
    print(string("    k₁(x,y) = ", formula_string(obj.kernel), " and ", argument_string(obj.kernel)))
end


*(a::Real, kernel::StandardMercerKernel) = ScaledMercerKernel(kernel, a)
*(kernel::StandardMercerKernel, a::Real) = a * kernel
*(a::Real, kernel::ScaledMercerKernel) = ScaledMercerKernel(deepcopy(kernel.kernel), a * kernel.a)
*(kernel::ScaledMercerKernel, a::Real) = (a * kernel.a) * deepcopy(kernel.kernel)


#== Product Kernel ====================#

type ProductMercerKernel <: MercerKernel
    a::Real
    lkernel::StandardMercerKernel
    rkernel::StandardMercerKernel
    function ProductMercerKernel(lkernel::StandardMercerKernel, rkernel::StandardMercerKernel, a::Real = 1)
        a > 0 || error("a = $a must be greater than zero.")
        new(a, lkernel, rkernel)
    end
end

function kernel(obj::ProductMercerKernel)
    k{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = convert(T, obj.a) * kernel(obj.lkernel)(x, y) * kernel(obj.rkernel)(x, y)
    return k
end


function show(io::IO, obj::ProductMercerKernel)
	println("Product Kernel: k(x,y) = ak₁(x,y)k₂(x,y) with a = $(obj.a)")
    println(string("    k₁(x,y) = ", formula_string(obj.lkernel), " and ", argument_string(obj.lkernel)))
    print(string("    k₂(x,y) = ", formula_string(obj.rkernel), " and ", argument_string(obj.rkernel)))
end


*(lkernel::StandardMercerKernel, rkernel::StandardMercerKernel) = ProductMercerKernel(lkernel, rkernel)
*(lkernel::ScaledMercerKernel, rkernel::StandardMercerKernel) = ProductMercerKernel(deepcopy(lkernel.kernel), rkernel, lkernel.a)
*(lkernel::StandardMercerKernel, rkernel::ScaledMercerKernel) = ProductMercerKernel(lkernel, deepcopy(rkernel.kernel), rkernel.a)
*(lkernel::ScaledMercerKernel, rkernel::ScaledMercerKernel) = ProductMercerKernel(deepcopy(lkernel.kernel), deepcopy(rkernel.kernel), lkernel.a * rkernel.a)


#===================================================================================================
  Instances of Mercer Kernel Types
===================================================================================================#

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
function kernel(obj::LinearKernel)
    if obj.c == 0
        k(x,y) = linearkernel(x, y)
    else
        k(x,y) = linearkernel(x, y, obj.c)
    end
    return k
end

formula_string(obj::LinearKernel) = "xᵗy + c"
argument_string(obj::LinearKernel) = "c = $(obj.c)"

function show(io::IO, obj::LinearKernel)
	print(string("Linear Kernel: k(x,y) = ", formula_string(obj), " with ", argument_string(obj)))
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
function kernel(obj::PolynomialKernel)
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

function show(io::IO, obj::PolynomialKernel)
	print(string("Polynomial Kernel: k(x,y) = ", formula_string(obj), " with ", argument_string(obj)))
end


#== Gaussian Kernel ===============#

function gaussiankernel{T<:FloatingPoint}(x::Array{T},y::Array{T},σ::Real)
	δ = x .- y
	return exp(- BLAS.dot(length(x), δ, 1, δ, 1) / (2*convert(T,σ)^2))
end

type GaussianKernel <: StandardMercerKernel
	σ::Real
	function GaussianKernel(σ::Real=1)
        σ >= 0 || error("σ = $(σ) must be greater than 0.")
		new(σ)
	end
end

arguments(obj::GaussianKernel) = obj.σ
function kernel(obj::GaussianKernel)
    k(x, y) = gaussiankernel(x, y, obj.σ)
    return k
end
    .
formula_string(obj::GaussianKernel) = "exp(-‖x-y‖²/(2σ²))"
argument_string(obj::GaussianKernel) = "σ = $(obj.σ)"

function show(io::IO,obj::GaussianKernel)
	print(string("Gaussian Kernel: k(x,y) = ", formula_string(obj), " with ", argument_string(obj)))
end


#== Exponential Kernel ===============#

function exponentialkernel{T<:FloatingPoint}(x::Array{T},y::Array{T},σ::Real)
	δ = x .- y
	return exp(- nrm2(length(x), δ, 1) / (2*convert(T,σ)^2) )
end

type ExponentialKernel <: MercerKernel
	k::Function
	σ::Real
	function ExponentialKernel(σ::Real=1)
		k(x,y) = exponentialkernel(x,y,σ)
		new(k,σ)
	end
end
function show(io::IO,obj::ExponentialKernel)
	print("Exponential Kernel: k(x,y) = exp(-‖x-y‖/(2σ²))   with σ = $(obj.σ)")
end

function sigmoidkernel{T<:FloatingPoint}(x::Array{T},y::Array{T},α::Real,c::Real)
	return tanh(convert(T,α)*BLAS.dot(length(x), x, 1, y, 1) + convert(T,c))
end
type SigmoidKernel <: MercerKernel
	k::Function
	α::Real
	c::Real
	function SigmoidKernel(α::Real=1,c::Real=0)
		k(x,y) = sigmoidkernel(x,y,α,c)
		new(k,α,c)
	end
end
function show(io::IO,obj::SigmoidKernel)
	print("Sigmoid Kernel: k(x,y) = tanh(α‖x-y‖² + c)   with α = $(obj.α) and c = $(obj.c)")
end



function rationalquadratickernel{T<:FloatingPoint}(x::Array{T},y::Array{T},c::Real)
	δ = x .- y
	dxy = BLAS.dot(length(x), δ, 1, δ, 1)
	return 1 - dxy/(dxy + convert(T,c))
end
type RationalQuadraticKernel <: MercerKernel
	k::Function
	c::Real
	function RationalQuadraticKernel(c::Real=1)
		k(x,y) = rationalquadratickernel(x,y,c)
		new(k,c)
	end
end
function show(io::IO,obj::RationalQuadraticKernel)
	print("Rational Quadratic Kernel: k(x,y) = 1 - ‖x-y‖²/(‖x-y‖² + c)   with c = $(obj.c)")
end


function multiquadratickernel{T<:FloatingPoint}(x::Array{T},y::Array{T},c::Real)
	δ = x .- y
	return sqrt(BLAS.dot(length(x), δ, 1, δ, 1) + convert(T,c)^2)
end
type MultiQuadraticKernel <: MercerKernel
	k::Function
	c::Real
	function MultiQuadraticKernel(c::Real=1)
		k(x,y) = multiquadratickernel(x,y,c)
		new(k,c)
	end
end
function show(io::IO,obj::MultiQuadraticKernel)
	print("Multi-Quadratic Kernel: k(x,y) = √(‖x-y‖² + c)   with c = $(obj.c)")
end


function inversemultiquadratickernel{T<:FloatingPoint}(x::Array{T},y::Array{T},c::Real)
	δ = x .- y
	return 1 / sqrt(BLAS.dot(length(x), δ, 1, δ, 1) + convert(T,c)^2)
end
type InverseMultiQuadraticKernel <: MercerKernel
	k::Function
	c::Real
	function InverseMultiQuadraticKernel(c::Real=1)
		k(x,y) = inversemultiquadratickernel(x,y,c)
		new(k,c)
	end
end
function show(io::IO,obj::InverseMultiQuadraticKernel)
	print("Inverse Multi-Quadratic Kernel: k(x,y) = 1/√(‖x-y‖² + c)   with c = $(obj.c)")
end


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
