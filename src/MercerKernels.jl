#===================================================================================================
  Mercer Kernel Type & Related Functions
===================================================================================================#

abstract MercerKernel

function kernel{T<:MercerKernel}(Kernel::T)
	return Kernel.k::Function
end

#===================================================================================================
  Instances of Mercer Kernel Types
===================================================================================================#

function linearkernel{T<:FloatingPoint}(x::Array{T},y::Array{T},c::Real)
	return BLAS.dot(length(x), x, 1, y, 1) + convert(T,c)
end

type LinearKernel <: MercerKernel
	k::Function
	c::Real
	function LinearKernel(c::Real=0)
		k(x,y) = linearkernel(x,y,c)
		new(k,c)
	end
end
function show(io::IO,obj::LinearKernel)
	print("Linear Kernel: k(x,y) = xᵗy + c   with c = $(obj.c)")
end


function polynomialkernel{T<:FloatingPoint}(x::Array{T},y::Array{T},α::Real,c::Real,d::Real)
	return (convert(T,α)*BLAS.dot(length(x), x, 1, y, 1) + convert(T,c))^convert(T,d)
end
type PolynomialKernel <: MercerKernel
	k::Function
	α::Real
	c::Real
	d::Real
	function PolynomialKernel(α::Real=1,c::Real=1,d::Real=2)
		k(x,y) = polynomialkernel(x,y,α,c,d)
		new(k,α,c,d)
	end
end
function show(io::IO,obj::PolynomialKernel)
	print("Polynomial Kernel: k(x,y) = (αxᵗy + c)ᵈ   with α = $(obj.α), c = $(obj.c) and d = $(obj.d)")
end


function gaussiankernel{T<:FloatingPoint}(x::Array{T},y::Array{T},σ::Real)
	δ = x .- y
	return exp(- BLAS.dot(length(x), δ, 1, δ, 1) / (2*convert(T,σ)^2))
end
type GaussianKernel <: MercerKernel
	k::Function
	σ::Real
	function GaussianKernel(σ::Real=1)
		k(x,y) = gaussiankernel(x,y,σ)
		new(k,σ)
	end
end
function show(io::IO,obj::GaussianKernel)
	print("Gaussian Kernel: k(x,y) = exp(-‖x-y‖²/(2σ²))   with σ = $(obj.σ)")
end


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
ale opinions here. His lying to you is fucked. He should never had lied. It's also fucked that he feels so ashamed about it that he feels the need to lie to you. He is obviously fearful, with good reason, of your reaction. Do you consider masturbating cheating? If it extends to that I think you both should get into counseling, preferably with someone who is sex positive, both separately and together. It's not healthy to deny someone a little self release now and again. And it really isn't healthy to have your self worth so adamantly tied into this black and white notion of cheating where the only sexual gratification your partner can get has to be directly linked to you every time and in every way.
I'm saying this, honestly, with completely warm intentions towards you. I'm not blasting you or anything like that. I get the impression that you are more bothered by the porn than the lying and that's worrisome. I think instead of focusing on the fact that he watched porn, you should instead focus on why he feels the need to hide it from you and the lying and what that sa

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


#===================================================================================================
  Kernel Matrix Functions
===================================================================================================#

# center_kernelmatrix!: Centralize a kernel matrix K
#	K := K - 1ₙ*K/n - K*1ₙ/n + 1ₙ*K*1ₙ/n^2
#	 • K is an n×n kernel matrix
#	 • 1ₙ is an n×n matrix of ones
function center_kernelmatrix!{T<:FloatingPoint}(K::Matrix{T})
	n = size(K,1)
	n == size(K,2) || error("Kernel matrix must be square")
	κ = sum(K,1)
	μₖ = sum(κ)/(convert(T,n)^2)
	BLAS.scal!(n,one(T)/convert(T,n),κ,1)
	return MATRIX.el_add!(MATRIX.col_add!(MATRIX.row_add!(K,-κ),-κ),μₖ)
end
center_kernelmatrix{T<:FloatingPoint}(K::Matrix{T}) = center_kernelmatrix!(copy(K))


# kernelmatrix:
#	Returns the kernel (Gramian) matrix K of X for mapping ϕ
#	 • X contains one observation per row
#	 • symmetrize = false will leave the bottom half of K without assigned values
function kernelmatrix{T₁<:FloatingPoint,T₂<:MercerKernel}(X::Matrix{T₁}, κᵩ::T₂=LinearKernel(); symmetrize::Bool=true)
	k = kernel(κᵩ)
	n = size(X,1)
	K = Array(T₁,n,n) # Kᵩ = XᵩXᵗᵩ
	for i = 1:n 
		for j = i:n
			K[i,j] = k(X[i,:],X[j,:])
		end 
	end
	return symmetrize ? MATRIX.syml!(K) : K
end


# kernelmatrix:
#	Returns the upper right corner kernel (Gramian) matrix K of [Xᵗ,Zᵗ]ᵗ
#	 • X is n×p and Z is m×p; contain one observation per row
#	 • Resulting matrix Kᵩ is n×m
function kernelmatrix{T₁<:FloatingPoint,T₂<:MercerKernel}(X::Matrix{T₁}, Z::Matrix{T₁}, κᵩ::T₂=LinearKernel())
	k = kernel(κᵩ)
	n,m = size(X,1), size(Z,1)
	size(X,2) == size(Z,2) || error("X ∈ ℝn×p and Z should be ∈ ℝm×p, but X ∈ ℝn×$(size(X,2)) and Z ∈ ℝm×$(size(Z,2)).")
	K = Array(T₁,n,m) # K = XᵩZᵗᵩ
	for j = 1:m 
		for i = 1:n
			K[i,j] = k(X[i,:],Z[j,:])
		end
	end
	return K
end

