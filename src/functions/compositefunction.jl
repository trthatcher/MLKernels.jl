#===================================================================================================
  Composition Classes: Valid kernel transformations
===================================================================================================#

abstract CompositionClass{T<:AbstractFloat}

@inline eltype{T}(::CompositionClass{T}) = T 

@inline iscomposable(::CompositionClass, ::RealFunction) = false

@inline ismercer(::CompositionClass) = false
@inline isnegdef(::CompositionClass) = false
@inline ismetric(::CompositionClass) = false
@inline isinnerprod(::CompositionClass) = false

@inline attainszero(::CompositionClass)     = true
@inline attainspositive(::CompositionClass) = true
@inline attainsnegative(::CompositionClass) = true

function description_string(g::CompositionClass, showtype::Bool = true)
    class = typeof(g)
    fields = fieldnames(class)
    class_str = string(class.name.name) * (showtype ? string("{", eltype(g), "}") : "")
    *(class_str, "(", join(["$field=$(getfield(g,field).value)" for field in fields], ","), ")")
end

function show(io::IO, g::CompositionClass)
    print(io, description_string(g))
end

function convert{T<:AbstractFloat,K<:CompositionClass}(::Type{CompositionClass{T}}, g::K)
    convert(K.name.primary{T}, g)
end

function =={T<:CompositionClass}(g1::T, g2::T)
    all([getfield(g1,field) == getfield(g2,field) for field in fieldnames(T)])
end

#== Positive Mercer Classes ==#

abstract PositiveMercerClass{T<:AbstractFloat} <: CompositionClass{T}
@inline ismercer(::PositiveMercerClass) = true
@inline attainsnegative(::PositiveMercerClass) = false
@inline attainszero(::PositiveMercerClass) = false

doc"GammaExponentialClass(α,γ) = exp(-α⋅fᵞ)   α ∈ (0,∞), γ ∈ (0,1]"
immutable GammaExponentialClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaExponentialClass(α::Variable{T}, γ::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(γ, Interval(Bound(zero(T), :open), Bound(one(T), :closed)))
    )
end
@outer_constructor(GammaExponentialClass, (1,0.5))
@inline iscomposable(::GammaExponentialClass, f::RealFunction) = isnegdef(f) && isnonnegative(f)
@inline composition{T<:AbstractFloat}(g::GammaExponentialClass{T}, z::T) = exp(-g.alpha * z^g.gamma)


doc"ExponentialClass(α) = exp(-α⋅f)   α ∈ (0,∞)"
immutable ExponentialClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::HyperParameter{T}
    ExponentialClass(α::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open))
    )
end
@outer_constructor(ExponentialClass, (1,))
@inline iscomposable(::ExponentialClass, f::RealFunction) = isnegdef(f) && isnonnegative(f)
@inline composition{T<:AbstractFloat}(g::ExponentialClass{T}, z::T) = exp(-g.alpha * z)


doc"GammaRationalClass(α,β,γ) = (1 + α⋅fᵞ)⁻ᵝ   α ∈ (0,∞), β ∈ (0,∞), γ ∈ (0,1]"
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
@inline iscomposable(::GammaRationalClass, f::RealFunction) = isnegdef(f) && isnonnegative(f)
@inline composition{T<:AbstractFloat}(g::GammaRationalClass{T}, z::T) = (1 + g.alpha*z^g.gamma)^(-g.beta)


doc"RationalClass(α,β) = (1 + α⋅f)⁻ᵝ   α ∈ (0,∞), β ∈ (0,∞)"
immutable RationalClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    RationalClass(α::Variable{T}, β::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(β, leftbounded(zero(T), :open))
    )
end
@outer_constructor(RationalClass, (1,1))
@inline iscomposable(::RationalClass, f::RealFunction) = isnegdef(f) && isnonnegative(f)
@inline composition{T<:AbstractFloat}(g::RationalClass{T}, z::T) = (1 + g.alpha*z)^(-g.beta)


doc"MatérnClass(ν,ρ) = 2ᵛ⁻¹(√(2ν)f/ρ)ᵛKᵥ(√(2ν)f/ρ)/Γ(ν)   ν ∈ (0,∞), ρ ∈ (0,∞)"
immutable MaternClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    nu::HyperParameter{T}
    rho::HyperParameter{T}
    MaternClass(ν::Variable{T}, ρ::Variable{T}) = new(
        HyperParameter(ν, leftbounded(zero(T), :open)),
        HyperParameter(ρ, leftbounded(zero(T), :open))
    )
end
@outer_constructor(MaternClass, (1,1))
@inline iscomposable(::MaternClass, f::RealFunction) = isnegdef(f) && isnonnegative(f)
@inline function composition{T<:AbstractFloat}(g::MaternClass{T}, z::T)
    v1 = sqrt(2g.nu) * z / g.rho
    v1 = v1 < eps(T) ? eps(T) : v1  # Overflow risk, z -> Inf
    2 * (v1/2)^(g.nu) * besselk(g.nu, v1) / gamma(g.nu)
end


doc"ExponentiatedClass(a,c) = exp(a⋅f + c)   a ∈ (0,∞), c ∈ [0,∞)"
immutable ExponentiatedClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    ExponentiatedClass(a::Variable{T}, c::Variable{T}) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed))
    )
end
@outer_constructor(ExponentiatedClass, (1,0))
@inline iscomposable(::ExponentiatedClass, f::RealFunction) = ismercer(f)
@inline composition{T<:AbstractFloat}(g::ExponentiatedClass{T}, z::T) = exp(g.a*z + g.c)


#== Other Mercer Classes ==#

doc"PolynomialClass(a,c,d) = (a⋅f + c)ᵈ   a ∈ (0,∞), c ∈ [0,∞), d ∈ ℤ+"
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
@inline iscomposable(::PolynomialClass, f::RealFunction) = ismercer(f)
@inline composition{T<:AbstractFloat}(g::PolynomialClass{T}, z::T) = (g.a*z + g.c)^g.d
@inline ismercer(::PolynomialClass) = true


#== Non-Negative Negative Definite Kernel Classes ==#

abstract NonNegNegDefClass{T<:AbstractFloat} <: CompositionClass{T}
@inline isnegdef(::NonNegNegDefClass) = true
@inline attainsnegative(::NonNegNegDefClass) = false

doc"PowerClass(a,c,γ) = (a⋅f + c)ᵞ   a ∈ (0,∞), c ∈ (0,∞), γ ∈ (0,1]"
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
@inline iscomposable(::PowerClass, f::RealFunction) = isnegdef(f) && isnonnegative(f)
@inline composition{T<:AbstractFloat}(g::PowerClass{T}, z::T) = (g.a*z + g.c)^(g.gamma)


doc"GammmaLogClass(α,γ) = log(1 + α⋅fᵞ)   α ∈ (0,∞), γ ∈ (0,1]"
immutable GammaLogClass{T<:AbstractFloat} <: NonNegNegDefClass{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaLogClass(α::Variable{T}, γ::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(γ, Interval(Bound(zero(T), :open), Bound(one(T), :closed)))
    )
end
@outer_constructor(GammaLogClass, (1,0.5))
@inline iscomposable(::GammaLogClass, f::RealFunction) = isnegdef(f) && isnonnegative(f)
@inline composition{T<:AbstractFloat}(g::GammaLogClass{T}, z::T) = log(g.alpha*z^(g.gamma) + 1)


doc"LogClass(α) = log(1 + α⋅f)   α ∈ (0,∞)"
immutable LogClass{T<:AbstractFloat} <: NonNegNegDefClass{T}
    alpha::HyperParameter{T}
    LogClass(α::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open))
    )
end
@outer_constructor(LogClass, (1,))
@inline iscomposable(::LogClass, f::RealFunction) = isnegdef(f) && isnonnegative(f)
@inline composition{T<:AbstractFloat}(g::LogClass{T}, z::T) = log(g.alpha*z + 1)


#== Non-Mercer, Non-Negative Definite Classes ==#

doc"SigmoidClass(a,c) = tanh(a⋅f + c)   a ∈ (0,∞), c ∈ (0,∞)"
immutable SigmoidClass{T<:AbstractFloat} <: CompositionClass{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    SigmoidClass(a::Variable{T}, c::Variable{T}) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed))   
    )
end
@outer_constructor(SigmoidClass, (1,0))
@inline iscomposable(::SigmoidClass, f::RealFunction) = ismercer(f)
@inline composition{T<:AbstractFloat}(g::SigmoidClass{T}, z::T) = tanh(g.a*z + g.c)


#===================================================================================================
  Kernel Composition ψ = g(f(x,y))
===================================================================================================#

doc"CompositeFunction(g,f) = g∘f   f:ℜⁿ×ℜⁿ→ℜ, g:ℜ→ℜ"
immutable CompositeFunction{T<:AbstractFloat} <: RealFunction{T}
    g::CompositionClass{T}
    f::PairwiseFunction{T}
    function CompositeFunction(g::CompositionClass{T}, f::PairwiseFunction{T})
        iscomposable(g, f) || error("Function is not composable.")
        new(g, f)
    end
end
function CompositeFunction{T<:AbstractFloat}(g::CompositionClass{T}, f::PairwiseFunction{T})
    CompositeFunction{T}(g, f)
end

∘(g::CompositionClass, f::PairwiseFunction) = CompositeFunction(g, f)

function convert{T<:AbstractFloat}(::Type{CompositeFunction{T}}, h::CompositeFunction)
    CompositeFunction(convert(CompositionClass{T}, h.g), convert(RealFunction{T}, h.f))
end

function description_string(f::CompositeFunction, showtype::Bool = true)
    obj_str = string("∘", showtype ? string("{", eltype(f), "}") : "")
    class_str = description_string(f.g, false)
    kernel_str = description_string(f.f, false)
    string(obj_str, "(g=", class_str, ",f=", kernel_str, ")")
end

==(h1::CompositeFunction, h2::CompositeFunction) = (h1.g == h2.g) && (h1.f == h2.f)

ismercer(h::CompositeFunction) = ismercer(h.g)
isnegdef(h::CompositeFunction) = isnegdef(h.g)

attainszero(h::CompositeFunction)     = attainszero(h.g)
attainspositive(h::CompositeFunction) = attainspositive(h.g)
attainsnegative(h::CompositeFunction) = attainsnegative(h.g)


#== Composition Kernels ==#

doc"GaussianKernel(α) = exp(-α⋅‖x-y‖²)"
function GaussianKernel{T<:AbstractFloat}(α::Variable{T})
    CompositeFunction(ExponentialClass(α), SquaredEuclidean{T}())
end
function GaussianKernel{T}(α::Argument{T}=1.0)
    GaussianKernel(Variable{T <: AbstractFloat ? T : Float64}(α))
end
SquaredExponentialKernel = GaussianKernel
RadialBasisKernel = GaussianKernel

doc"LaplacianKernel(α) = exp(α⋅‖x-y‖)"
function LaplacianKernel{T<:AbstractFloat}(α::Variable{T})
    CompositeFunction(GammaExponentialClass(α, Variable{T}(0.5)), SquaredEuclidean{T}())
end
function LaplacianKernel{T<:Real}(α::Argument{T}=1.0)
    LaplacianKernel(Variable{T <: AbstractFloat ? T : Float64}(α))
end

doc"PeriodicKernel(α,p) = exp(-α⋅Σⱼsin²(p(xⱼ-yⱼ)))"
function PeriodicKernel{T<:AbstractFloat}(α::Variable{T}, p::Variable{T})
    CompositeFunction(ExponentialClass(α), SineSquaredKernel(p))
end
function PeriodicKernel{T1<:Real,T2<:Real}(
        α::Argument{T1} = 1.0,
        p::Argument{T2} = convert(T1 <: AbstractFloat ? T1 : Float64,π)
    )
    Tmax = promote_type(T1, T2)
    T = Tmax <: AbstractFloat ? Tmax : Float64
    PeriodicKernel(Variable{T}(α), Variable{T}(p))
end

doc"RationalQuadraticKernel(α,β) = (1 + α⋅‖x-y‖²)⁻ᵝ"
function RationalQuadraticKernel{T<:AbstractFloat}(α::Variable{T}, β::Variable{T})
    CompositeFunction(RationalClass(α, β), SquaredEuclidean{T}())
end
function RationalQuadraticKernel{T1<:Real,T2<:Real}(
        α::Argument{T1} = 1.0,
        β::Argument{T2} =one(T1)
    )
    Tmax = promote_type(T1, T2)
    T = Tmax <: AbstractFloat ? Tmax : Float64
    RationalQuadraticKernel(Variable{T}(α), Variable{T}(β))
end

doc"MatérnKernel(ν,θ) = 2ᵛ⁻¹(√(2ν)‖x-y‖²/θ)ᵛKᵥ(√(2ν)‖x-y‖²/θ)/Γ(ν)"
function MaternKernel{T<:AbstractFloat}(ν::Variable{T}, θ::Variable{T})
    CompositeFunction(MaternClass(ν, θ), SquaredEuclidean{T}())
end
function MaternKernel{T1<:Real,T2<:Real}(ν::Argument{T1} = 1.0, θ::Argument{T2} = one(T1))
    Tmax = promote_type(T1, T2)
    T = Tmax <: AbstractFloat ? Tmax : Float64
    MaternKernel(Variable{T}(ν), Variable{T}(θ))
end

MatérnKernel = MaternKernel

doc"PolynomialKernel(a,c,d) = (a⋅xᵀy + c)ᵈ"
function PolynomialKernel{T<:AbstractFloat,U<:Integer}(
        a::Argument{T} = 1.0,
        c::Argument{T} = one(T),
        d::Argument{U} = 3
    )
    CompositeFunction(PolynomialClass(a, c, d), ScalarProduct{T}())
end

doc"LinearKernel(α,c,d) = a⋅xᵀy + c"
function LinearKernel{T<:AbstractFloat}(a::Argument{T} = 1.0, c::Argument{T} = one(T))
    CompositeFunction(PolynomialClass(a, c, 1), ScalarProduct{T}())
end

doc"SigmoidKernel(α,c) = tanh(a⋅xᵀy + c)"
function SigmoidKernel{T<:Real}(a::Argument{T} = 1.0, c::Argument{T} = one(T))
    CompositeFunction(SigmoidClass(a, c), ScalarProduct{T}())
end


#== Special Compositions ==#

function ^{T<:AbstractFloat}(f::PairwiseFunction{T}, d::Integer)
    CompositeFunction(PolynomialClass(one(T), zero(T), d), f)
end

function ^{T<:AbstractFloat}(f::PairwiseFunction{T}, γ::T)
    CompositeFunction(PowerClass(one(T), zero(T), γ), f)
end

function exp{T<:AbstractFloat}(f::PairwiseFunction{T})
    CompositeFunction(ExponentiatedClass(one(T), zero(T)), f)
end

function tanh{T<:AbstractFloat}(f::PairwiseFunction{T})
    CompositeFunction(SigmoidClass(one(T), zero(T)), f)
end
