#===================================================================================================
  Composition Classes: Valid kernel transformations
===================================================================================================#

abstract CompositionClass{T<:AbstractFloat}

@inline eltype{T}(::CompositionClass{T}) = T 

@inline iscomposable(::CompositionClass, ::RealKernel) = false

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
@inline iscomposable(::GammaExponentialClass, f::RealKernel) = isnegdef(f) && isnonnegative(f)
@inline composition{T<:AbstractFloat}(g::GammaExponentialClass{T}, z::T) = exp(-g.alpha * z^g.gamma)


doc"ExponentialClass(α) = exp(-α⋅f)   α ∈ (0,∞)"
immutable ExponentialClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::HyperParameter{T}
    ExponentialClass(α::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open))
    )
end
@outer_constructor(ExponentialClass, (1,))
@inline iscomposable(::ExponentialClass, f::RealKernel) = isnegdef(f) && isnonnegative(f)
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
@inline iscomposable(::GammaRationalClass, f::RealKernel) = isnegdef(f) && isnonnegative(f)
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
@inline iscomposable(::RationalClass, f::RealKernel) = isnegdef(f) && isnonnegative(f)
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
@inline iscomposable(::MaternClass, f::RealKernel) = isnegdef(f) && isnonnegative(f)
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
@inline iscomposable(::ExponentiatedClass, f::RealKernel) = ismercer(f)
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
@inline iscomposable(::PolynomialClass, f::RealKernel) = ismercer(f)
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
@inline iscomposable(::PowerClass, f::RealKernel) = isnegdef(f) && isnonnegative(f)
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
@inline iscomposable(::GammaLogClass, f::RealKernel) = isnegdef(f) && isnonnegative(f)
@inline composition{T<:AbstractFloat}(g::GammaLogClass{T}, z::T) = log(g.alpha*z^(g.gamma) + 1)


doc"LogClass(α) = log(1 + α⋅f)   α ∈ (0,∞)"
immutable LogClass{T<:AbstractFloat} <: NonNegNegDefClass{T}
    alpha::HyperParameter{T}
    LogClass(α::Variable{T}) = new(
        HyperParameter(α, leftbounded(zero(T), :open))
    )
end
@outer_constructor(LogClass, (1,))
@inline iscomposable(::LogClass, f::RealKernel) = isnegdef(f) && isnonnegative(f)
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
@inline iscomposable(::SigmoidClass, f::RealKernel) = ismercer(f)
@inline composition{T<:AbstractFloat}(g::SigmoidClass{T}, z::T) = tanh(g.a*z + g.c)


#===================================================================================================
  Kernel Composition ψ = g(f(x,y))
===================================================================================================#

doc"CompositeKernel(g,f) = g∘f   f:ℝⁿ×ℝⁿ→ℝ, g:ℝ→ℝ"
immutable CompositeKernel{T<:AbstractFloat} <: RealKernel{T}
    g::CompositionClass{T}
    f::PairwiseKernel{T}
    function CompositeKernel(g::CompositionClass{T}, f::PairwiseKernel{T})
        iscomposable(g, f) || error("Kernel is not composable.")
        new(g, f)
    end
end
function CompositeKernel{T<:AbstractFloat}(g::CompositionClass{T}, f::PairwiseKernel{T})
    CompositeKernel{T}(g, f)
end

∘(g::CompositionClass, f::PairwiseKernel) = CompositeKernel(g, f)

function convert{T<:AbstractFloat}(::Type{CompositeKernel{T}}, h::CompositeKernel)
    CompositeKernel(convert(CompositionClass{T}, h.g), convert(RealKernel{T}, h.f))
end

function description_string(f::CompositeKernel, showtype::Bool = true)
    obj_str = string("Composition", showtype ? string("{", eltype(f), "}") : "")
    class_str = description_string(f.g, false)
    kernel_str = description_string(f.f, false)
    string(obj_str, "(g=", class_str, ",f=", kernel_str, ")")
end

==(h1::CompositeKernel, h2::CompositeKernel) = (h1.g == h2.g) && (h1.f == h2.f)

ismercer(h::CompositeKernel) = ismercer(h.g)
isnegdef(h::CompositeKernel) = isnegdef(h.g)

attainszero(h::CompositeKernel)     = attainszero(h.g)
attainspositive(h::CompositeKernel) = attainspositive(h.g)
attainsnegative(h::CompositeKernel) = attainsnegative(h.g)


#== Composition Kernels ==#

doc"GaussianKernel(α) = exp(-α⋅‖x-y‖²)"
function GaussianKernel{T1<:Real}(α::Argument{T1} = 1.0)
    T = promote_type_float(T1)
    CompositeKernel(ExponentialClass(Variable{T}(α)), SquaredEuclidean{T}())
end
SquaredExponentialKernel = GaussianKernel
RadialBasisKernel = GaussianKernel

doc"LaplacianKernel(α) = exp(α⋅‖x-y‖)"
function LaplacianKernel{T1<:Real}(α::Argument{T1} = 1.0)
    T = promote_type_float(T1)
    CompositeKernel(GammaExponentialClass(Variable{T}(α), Variable{T}(0.5)),
                      SquaredEuclidean{T}())
end

doc"PeriodicKernel(α,p) = exp(-α⋅Σⱼsin²(p(xⱼ-yⱼ)))"
function PeriodicKernel{T1<:Real,T2<:Real}(
        α::Argument{T1} = 1.0,
        p::Argument{T2} = convert(promote_type_float(T1), π)
    )
    T = promote_type_float(T1, T2)
    CompositeKernel(ExponentialClass(Variable{T}(α)), SineSquaredKernel(Variable{T}(p)))
end

doc"RationalQuadraticKernel(α,β) = (1 + α⋅‖x-y‖²)⁻ᵝ"
function RationalQuadraticKernel{T1<:Real,T2<:Real}(
        α::Argument{T1} = 1.0,
        β::Argument{T2} = one(T1)
    )
    T = promote_type_float(T1, T2)
    CompositeKernel(RationalClass(Variable{T}(α), Variable{T}(β)), SquaredEuclidean{T}())
end

doc"MatérnKernel(ν,θ) = 2ᵛ⁻¹(√(2ν)‖x-y‖²/θ)ᵛKᵥ(√(2ν)‖x-y‖²/θ)/Γ(ν)"
function MaternKernel{T1<:Real,T2<:Real}(
        ν::Argument{T1} = 1.0,
        θ::Argument{T2} = one(T1)
    )
    T = promote_type_float(T1, T2)
    CompositeKernel(MaternClass(Variable{T}(ν), Variable{T}(θ)), SquaredEuclidean{T}())
end
MatérnKernel = MaternKernel

doc"PolynomialKernel(a,c,d) = (a⋅xᵀy + c)ᵈ"
function PolynomialKernel{T1<:Real,T2<:Real,U<:Integer}(
        a::Argument{T1} = 1.0,
        c::Argument{T2} = one(T1),
        d::Argument{U} = 3
    )
    T = promote_type_float(T1, T2)
    CompositeKernel(PolynomialClass(Variable{T}(a), Variable{T}(c), Variable{U}(d)),
                      ScalarProduct{T}())
end

doc"LinearKernel(α,c,d) = a⋅xᵀy + c"
function LinearKernel{T1<:Real,T2<:Real}(
        a::Argument{T1} = 1.0,
        c::Argument{T2} = one(T1)
    )
    T = promote_type_float(T1, T2)
    CompositeKernel(PolynomialClass(Variable{T}(a), Variable{T}(c), 1), ScalarProduct{T}())
end

doc"SigmoidKernel(α,c) = tanh(a⋅xᵀy + c)"
function SigmoidKernel{T1<:Real,T2<:Real}(
        a::Argument{T1} = 1.0,
        c::Argument{T2} = one(T1)
    )
    T = promote_type_float(T1, T2)
    CompositeKernel(SigmoidClass(Variable{T}(a), Variable{T}(c)), ScalarProduct{T}())
end


#== Special Compositions ==#

function ^{T<:AbstractFloat}(f::PairwiseKernel{T}, d::Integer)
    CompositeKernel(PolynomialClass(one(T), zero(T), d), f)
end

function ^{T<:AbstractFloat}(f::PairwiseKernel{T}, γ::T)
    CompositeKernel(PowerClass(one(T), zero(T), γ), f)
end

function exp{T<:AbstractFloat}(f::PairwiseKernel{T})
    CompositeKernel(ExponentiatedClass(one(T), zero(T)), f)
end

function tanh{T<:AbstractFloat}(f::PairwiseKernel{T})
    CompositeKernel(SigmoidClass(one(T), zero(T)), f)
end
