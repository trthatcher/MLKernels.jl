doc"GaussianKernel(α) = exp(-α⋅‖x-y‖²)"
function GaussianKernel{T<:AbstractFloat}(α::Argument{T} = 1.0)
    KernelComposition(ExponentialClass(α), SquaredDistanceKernel{T}())
end
SquaredExponentialKernel = GaussianKernel
RadialBasisKernel = GaussianKernel

doc"LaplacianKernel(α) = exp(α⋅‖x-y‖)"
function LaplacianKernel{T<:AbstractFloat}(α::Argument{T} = 1.0)
    KernelComposition(GammaExponentialClass(α, convert(T, 0.5)), SquaredDistanceKernel{T}())
end

doc"PeriodicKernel(α,p) = exp(-α⋅Σⱼsin²(p(xⱼ-yⱼ)))"
function PeriodicKernel{T<:AbstractFloat}(α::Argument{T} = 1.0, p::Argument{T} = convert(T, π))
    KernelComposition(ExponentialClass(α), SineSquaredKernel(p))
end

doc"RationalQuadraticKernel(α,β) = (1 + α⋅‖x-y‖²)⁻ᵝ"
function RationalQuadraticKernel{T<:Real}(α::Argument{T} = 1.0, β::Argument{T} = one(T))
    KernelComposition(RationalClass(α, β), SquaredDistanceKernel{T}())
end

doc"MatérnKernel(ν,θ) = 2ᵛ⁻¹(√(2ν)‖x-y‖²/θ)ᵛKᵥ(√(2ν)‖x-y‖²/θ)/Γ(ν)"
function MaternKernel{T<:AbstractFloat}(ν::Argument{T} = 1.0, θ::Argument{T} = one(T))
    KernelComposition(MaternClass(ν, θ), SquaredDistanceKernel{T}())
end
MatérnKernel = MaternKernel

doc"PolynomialKernel(a,c,d) = (a⋅xᵀy + c)ᵈ"
function PolynomialKernel{T<:AbstractFloat,U<:Integer}(
        a::Argument{T} = 1.0,
        c::Argument{T} = one(T),
        d::Argument{U} = 3
    )
    KernelComposition(PolynomialClass(a, c, d), ScalarProductKernel{T}())
end

doc"LinearKernel(α,c,d) = a⋅xᵀy + c"
function LinearKernel{T<:AbstractFloat}(a::Argument{T} = 1.0, c::Argument{T} = one(T))
    KernelComposition(PolynomialClass(a, c, 1), ScalarProductKernel{T}())
end

doc"SigmoidKernel(α,c) = tanh(a⋅xᵀy + c)"
function SigmoidKernel{T<:Real}(a::Argument{T} = 1.0, c::Argument{T} = one(T))
    KernelComposition(SigmoidClass(a, c), ScalarProductKernel{T}())
end
