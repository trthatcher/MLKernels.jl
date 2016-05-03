#===================================================================================================
  Composition Classes: Valid kernel transformations
===================================================================================================#

abstract CompositionClass{T<:AbstractFloat}

@inline eltype{T}(::CompositionClass{T}) = T 

@inline iscomposable(::CompositionClass, ::Kernel) = false

@inline ismercer(::CompositionClass) = false
@inline isnegdef(::CompositionClass) = false

@inline attainszero(::CompositionClass)     = true
@inline attainspositive(::CompositionClass) = true
@inline attainsnegative(::CompositionClass) = true

function description_string(ϕ::CompositionClass, showtype::Bool = true)
    class = typeof(ϕ)
    fields = fieldnames(class)
    class_str = string(class.name.name) * (showtype ? string("{", eltype(ϕ), "}") : "")
    *(class_str, "(", join(["$field=$(getfield(ϕ,field).value)" for field in fields], ","), ")")
end

function show(io::IO, ϕ::CompositionClass)
    print(io, description_string(ϕ))
end

function convert{T<:AbstractFloat,K<:CompositionClass}(::Type{CompositionClass{T}}, ϕ::K)
    convert(K.name.primary{T}, ϕ)
end

function =={T<:CompositionClass}(ϕ1::T, ϕ2::T)
    all([getfield(ϕ1,field) == getfield(ϕ2,field) for field in fieldnames(T)])
end

#== Positive Mercer Classes ==#

abstract PositiveMercerClass{T<:AbstractFloat} <: CompositionClass{T}
@inline ismercer(::PositiveMercerClass) = true
@inline attainsnegative(::PositiveMercerClass) = false
@inline attainszero(::PositiveMercerClass) = false

doc"GammaExponentialClass(κ;α,γ) = exp(-α⋅κᵞ)"
immutable GammaExponentialClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaExponentialClass(α::Variable{T}, γ::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(γ, Interval(Bound(zero(T), :open), Bound(one(T), :closed)))
    )
end
@outer_constructor(GammaExponentialClass, (1,0.5))
@inline iscomposable(::GammaExponentialClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::GammaExponentialClass{T}, z::T) = exp(-ϕ.alpha * z^ϕ.gamma)


doc"ExponentialClass(κ;α) = exp(-α⋅κ²)"
immutable ExponentialClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::HyperParameter{T}
    ExponentialClass(α::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open))
    )
end
@outer_constructor(ExponentialClass, (1,))
@inline iscomposable(::ExponentialClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::ExponentialClass{T}, z::T) = exp(-ϕ.alpha * z)


doc"GammaRationalClass(κ;α,β,γ) = (1 + α⋅κᵞ)⁻ᵝ"
immutable GammaRationalClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaRationalClass(α::Variable{T}, β::Variable{T}, γ::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(β, leftbounded(zero(T), :open)),
        HyperParameter(γ, Interval(Bound(zero(T), :open), Bound(one(T), :closed)))
    )
end
@outer_constructor(GammaRationalClass, (1,1,0.5))
@inline iscomposable(::GammaRationalClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::GammaRationalClass{T}, z::T) = (1 + ϕ.alpha*z^ϕ.gamma)^(-ϕ.beta)


doc"RationalClass(κ;α,β,γ) = (1 + α⋅κ)⁻ᵝ"
immutable RationalClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    RationalClass(α::Variable{T}, β::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(β, leftbounded(zero(T), :open))
    )
end
@outer_constructor(RationalClass, (1,1))
@inline iscomposable(::RationalClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::RationalClass{T}, z::T) = (1 + ϕ.alpha*z)^(-ϕ.beta)


doc"MatérnClass(κ;ν,ρ) = 2ᵛ⁻¹(√(2ν)κ/ρ)ᵛKᵥ(√(2ν)κ/ρ)/Γ(ν)"
immutable MaternClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    nu::HyperParameter{T}
    rho::HyperParameter{T}
    MaternClass(ν::Variable{T}, ρ::Variable{T}) = new(
        HyperParameter(ν, leftbounded(zero(T), :open)),
        HyperParameter(ρ, leftbounded(zero(T), :open))
    )
end
@outer_constructor(MaternClass, (1,1))
@inline iscomposable(::MaternClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline function phi{T<:AbstractFloat}(ϕ::MaternClass{T}, z::T)
    v1 = sqrt(2ϕ.nu) * z / ϕ.rho
    v1 = v1 < eps(T) ? eps(T) : v1  # Overflow risk, z -> Inf
    2 * (v1/2)^(ϕ.nu) * besselk(ϕ.nu, v1) / gamma(ϕ.nu)
end


doc"ExponentiatedClass(κ;α) = exp(a⋅κ + c)"
immutable ExponentiatedClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    ExponentiatedClass(a::Variable{T}, c::Variable{T}) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed))
    )
end
@outer_constructor(ExponentiatedClass, (1,0))
@inline iscomposable(::ExponentiatedClass, κ::Kernel) = ismercer(κ)
@inline phi{T<:AbstractFloat}(ϕ::ExponentiatedClass{T}, z::T) = exp(ϕ.a*z + ϕ.c)


#== Other Mercer Classes ==#

doc"PolynomialClass(κ;a,c,d) = (a⋅κ + c)ᵈ"
immutable PolynomialClass{T<:AbstractFloat,U<:Integer} <: CompositionClass{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    d::HyperParameter{U}
    PolynomialClass(a::Variable{T}, c::Variable{T}, d::Variable{U}) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed)),
        HyperParameter(d, leftbounded(one(U),  :closed))
    )
end
@outer_constructor(PolynomialClass, (1,0,3))
@inline iscomposable(::PolynomialClass, κ::Kernel) = ismercer(κ)
@inline phi{T<:AbstractFloat}(ϕ::PolynomialClass{T}, z::T) = (ϕ.a*z + ϕ.c)^ϕ.d
@inline ismercer(::PolynomialClass) = true


#== Non-Negative Negative Definite Kernel Classes ==#

abstract NonNegNegDefClass{T<:AbstractFloat} <: CompositionClass{T}
@inline isnegdef(::NonNegNegDefClass) = true
@inline attainsnegative(::NonNegNegDefClass) = false

doc"PowerClass(z;a,c,γ) = (az + c)ᵞ"
immutable PowerClass{T<:AbstractFloat} <: NonNegNegDefClass{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    gamma::HyperParameter{T}
    PowerClass(a::Variable{T}, c::Variable{T}, γ::Variable{T}) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed)),
        HyperParameter(γ, Interval(Bound(zero(T), :open), Bound(one(T), :closed)))
    )
end
@outer_constructor(PowerClass, (1,0,0.5))
@inline iscomposable(::PowerClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::PowerClass{T}, z::T) = (ϕ.a*z + ϕ.c)^(ϕ.gamma)


doc"GammmaLogClass(z;α,γ) = log(1 + α⋅zᵞ)"
immutable GammaLogClass{T<:AbstractFloat} <: NonNegNegDefClass{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaLogClass(α::Variable{T}, γ::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(γ, Interval(Bound(zero(T), :open), Bound(one(T), :closed)))
    )
end
@outer_constructor(GammaLogClass, (1,0.5))
@inline iscomposable(::GammaLogClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::GammaLogClass{T}, z::T) = log(ϕ.alpha*z^(ϕ.gamma) + 1)


doc"LogClass(z;α) = log(1 + α⋅z)"
immutable LogClass{T<:AbstractFloat} <: NonNegNegDefClass{T}
    alpha::HyperParameter{T}
    LogClass(α::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open))
    )
end
@outer_constructor(LogClass, (1,))
@inline iscomposable(::LogClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::LogClass{T}, z::T) = log(ϕ.alpha*z + 1)


#== Non-Mercer, Non-Negative Definite Classes ==#

doc"SigmoidClass(κ;α,c) = tanh(a⋅κ + c)"
immutable SigmoidClass{T<:AbstractFloat} <: CompositionClass{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    SigmoidClass(a::Variable{T}, c::Variable{T}) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed))   
    )
end
@outer_constructor(SigmoidClass, (1,0))
@inline iscomposable(::SigmoidClass, κ::Kernel) = ismercer(κ)
@inline phi{T<:AbstractFloat}(ϕ::SigmoidClass{T}, z::T) = tanh(ϕ.a*z + ϕ.c)


#===================================================================================================
  Kernel Composition ψ = ϕ(κ(x,y))
===================================================================================================#

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

function convert{T<:AbstractFloat}(::Type{KernelComposition{T}}, κ::KernelComposition)
    KernelComposition(convert(CompositionClass{T}, κ.phi), convert(Kernel{T}, κ.kappa))
end

function description_string(κ::KernelComposition, showtype::Bool = true)
    obj_str = string("KernelComposition", showtype ? string("{", eltype(κ), "}") : "")
    class_str = description_string(κ.phi, false)
    kernel_str = description_string(κ.kappa, false)
    string(obj_str, "(phi=", class_str, ",kappa=", kernel_str, ")")
end

function ==(ψ1::KernelComposition, ψ2::KernelComposition)
    (ψ1.phi == ψ2.phi) && (ψ1.kappa == ψ2.kappa)
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
