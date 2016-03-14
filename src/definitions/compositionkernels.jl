doc"GaussianKernel(α) = exp(-α⋅‖x-y‖²)"
function GaussianKernel(α::Variable = 1)
    θ = promote_arguments(Float64, α, 1, 1)
    KernelComposition(ExponentialClass(θ[1:2]...), SquaredDistanceKernel(θ[3]))
end
SquaredExponentialKernel = GaussianKernel
RadialBasisKernel = GaussianKernel

doc"LaplacianKernel(α) = exp(α⋅‖x-y‖)"
function LaplacianKernel(α::Variable = 1)
    θ = promote_arguments(Float64, α, 1//2, 1)
    KernelComposition(ExponentialClass(θ[1:2]...), SquaredDistanceKernel(θ[3]))
end

doc"PeriodicKernel(α,p) = exp(-α⋅Σⱼsin²(p(xⱼ-yⱼ)))"
function PeriodicKernel{T<:Real}(α::T = 1.0, p::Real = convert(T, π))
    U = promote_type(T, typeof(p))
    U = T <: AbstractFloat ? T : Float64
    KernelComposition(ExponentialClass(convert(U, α), one(T)), SineSquaredKernel(convert(U, p), 
                      one(T)))
end

doc"RationalQuadraticKernel(α,β) = (1 + α⋅‖x-y‖²)⁻ᵝ"
function RationalQuadraticKernel{T<:Real}(α::T = 1.0, β::Real = one(T))
    U = promote_type(T, typeof(β))
    U = T <: AbstractFloat ? T : Float64
    KernelComposition(RationalQuadraticClass(convert(U, α), convert(U, β)), 
                      SquaredDistanceKernel(one(T)))
end

doc"MatérnKernel(ν,θ) = 2ᵛ⁻¹(√(2ν)‖x-y‖²/θ)ᵛKᵥ(√(2ν)‖x-y‖²/θ)/Γ(ν)"
function MaternKernel{T<:Real}(ν::T = 1.0, θ::Real = one(T))
    U = promote_type(T, typeof(θ))
    U = T <: AbstractFloat ? T : Float64
    KernelComposition(MaternClass(convert(U, ν), convert(U, θ)), SquaredDistanceKernel(one(U)))
end
MatérnKernel = MaternKernel

doc"PolynomialKernel(a,c,d) = (a⋅xᵀy + c)ᵈ"
function PolynomialKernel{T<:Real}(a::T = 1.0, c::Real = one(T), d::Real = 3one(T))
    U = promote_type(T, typeof(c), typeof(d))
    U = T <: AbstractFloat ? T : Float64
    KernelComposition(PolynomialClass(convert(U, a), convert(U, c), convert(U, d)), 
                      ScalarProductKernel{U}())
end

doc"LinearKernel(α,c,d) = a⋅xᵀy + c"
function LinearKernel{T<:Real}(a::T = 1.0, c::Real = one(T))
    U = promote_type(T, typeof(c))
    U = T <: AbstractFloat ? T : Float64
    KernelComposition(PolynomialClass(convert(U, a), convert(U, c), one(U)), 
                      ScalarProductKernel{U}())
end

doc"SigmoidKernel(α,c) = tanh(a⋅xᵀy + c)"
function SigmoidKernel{T<:Real}(a::T = 1.0, c::Real = one(T))
    U = promote_type(T, typeof(c))
    U = T <: AbstractFloat ? T : Float64
    KernelComposition(SigmoidClass(convert(U, a), convert(U, c)), ScalarProductKernel{U}())
end
