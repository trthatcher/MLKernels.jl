#==========================================================================
  Kernel Composition ψ = ϕ(κ(x,y))
==========================================================================#

doc"KernelComposition(ϕ,κ) = ϕ∘κ"
immutable KernelComposition{T<:AbstractFloat} <: StandardKernel{T}
    phi::CompositionClass{T}
    kappa::PairwiseKernel{T}
    function KernelComposition(ϕ::CompositionClass{T}, κ::PairwiseKernel{T})
        iscomposable(ϕ, κ) || error("Kernel is not composable.")
        new(ϕ, κ)
    end
end
function KernelComposition{T<:AbstractFloat}(ϕ::CompositionClass{T}, κ::PairwiseKernel{T})
    KernelComposition{T}(ϕ, κ)
end

function description_string(κ::KernelComposition)
    string("∘(",description_string(κ.phi), ",", description_string(κ.kappa),")")
end

ismercer(κ::KernelComposition) = ismercer(κ.phi)
isnegdef(κ::KernelComposition) = isnegdef(κ.phi)

attainszero(κ::KernelComposition)     = attainszero(κ.phi)
attainspositive(κ::KernelComposition) = attainspositive(κ.phi)
attainsnegative(κ::KernelComposition) = attainsnegative(κ.phi)


#== Composition Kernels ==#

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
function RationalQuadraticKernel{T<:AbstractFloat}(α::Argument{T} = 1.0, β::Argument{T} = one(T))
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


#== Special Compositions ==#

∘(ϕ::CompositionClass, κ::Kernel) = KernelComposition(ϕ, κ)

function ^{T<:AbstractFloat}(κ::PairwiseKernel{T}, d::Integer)
    KernelComposition(PolynomialClass(one(T), zero(T), d), κ)
end

function ^{T<:AbstractFloat}(κ::PairwiseKernel{T}, γ::T)
    KernelComposition(PowerClass(one(T), zero(T), γ), κ)
end

function exp{T<:AbstractFloat}(κ::PairwiseKernel{T})
    KernelComposition(ExponentiatedClass(one(T), zero(T)), κ)
end

function tanh{T<:AbstractFloat}(κ::PairwiseKernel{T})
    KernelComposition(SigmoidClass(one(T), zero(T)), κ)
end
