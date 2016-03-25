#==========================================================================
  Composition Classes
==========================================================================#

#== Positive Mercer Classes ==#

abstract PositiveMercerClass{T<:AbstractFloat} <: CompositionClass{T}

doc"GammaExponentialClass(κ;α,γ) = exp(-α⋅κᵞ)"
immutable GammaExponentialClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::Parameter{T}
    gamma::Parameter{T}
    GammaExponentialClass(α::Variable{T}, γ::Variable{T}) = new(
        Parameter(α, LowerBound(zero(T), :strict)),
        Parameter(γ, Interval(Bound(zero(T), :strict), Bound(one(T), :nonstrict)))
    )
end
@outer_constructor(GammaExponentialClass, (1,0.5))
@inline iscomposable(::GammaExponentialClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::GammaExponentialClass{T}, z::T) = exp(-ϕ.alpha * z^ϕ.gamma)


doc"ExponentialClass(κ;α) = exp(-α⋅κ²)"
immutable ExponentialClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::Parameter{T}
    ExponentialClass(α::Variable{T}) = new(
        Parameter(α, LowerBound(zero(T), :strict))
    )
end
@outer_constructor(ExponentialClass, (1,))
@inline iscomposable(::ExponentialClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::ExponentialClass{T}, z::T) = exp(-ϕ.alpha * z)


doc"GammaRationalClass(κ;α,β,γ) = (1 + α⋅κᵞ)⁻ᵝ"
immutable GammaRationalClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::Parameter{T}
    beta::Parameter{T}
    gamma::Parameter{T}
    GammaRationalClass(α::Variable{T}, β::Variable{T}, γ::Variable{T}) = new(
        Parameter(α, LowerBound(zero(T), :strict)),
        Parameter(β, LowerBound(zero(T), :strict)),
        Parameter(γ, Interval(Bound(zero(T), :strict), Bound(one(T), :nonstrict)))
    )
end
@outer_constructor(GammaRationalClass, (1,1,0.5))
@inline iscomposable(::GammaRationalClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::GammaRationalClass{T}, z::T) = (1 + ϕ.alpha*z^ϕ.gamma)^(-ϕ.beta)


doc"RationalClass(κ;α,β,γ) = (1 + α⋅κ)⁻ᵝ"
immutable RationalClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::Parameter{T}
    beta::Parameter{T}
    RationalClass(α::Variable{T}, β::Variable{T}) = new(
        Parameter(α, LowerBound(zero(T), :strict)),
        Parameter(β, LowerBound(zero(T), :strict))
    )
end
@outer_constructor(RationalClass, (1,1))
@inline iscomposable(::RationalClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::RationalClass{T}, z::T) = (1 + ϕ.alpha*z)^(-ϕ.beta)


doc"MatérnClass(κ;ν,ρ) = 2ᵛ⁻¹(√(2ν)κ/ρ)ᵛKᵥ(√(2ν)κ/ρ)/Γ(ν)"
immutable MaternClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    nu::Parameter{T}
    rho::Parameter{T}
    MaternClass(ν::Variable{T}, ρ::Variable{T}) = new(
        Parameter(ν, LowerBound(zero(T), :strict)),
        Parameter(ρ, LowerBound(zero(T), :strict))
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
    a::Parameter{T}
    c::Parameter{T}
    ExponentiatedClass(a::Variable{T}, c::Variable{T}) = new(
        Parameter(a, LowerBound(zero(T), :strict)),
        Parameter(c, LowerBound(zero(T), :nonstrict))
    )
end
@outer_constructor(ExponentiatedClass, (1,0))
@inline iscomposable(::ExponentiatedClass, κ::Kernel) = ismercer(κ)
@inline phi{T<:AbstractFloat}(ϕ::ExponentiatedClass{T}, z::T) = exp(ϕ.a*z + ϕ.c)


#== Other Mercer Classes ==#

doc"PolynomialClass(κ;a,c,d) = (a⋅κ + c)ᵈ"
immutable PolynomialClass{T<:AbstractFloat,U<:Integer} <: CompositionClass{T}
    a::Parameter{T}
    c::Parameter{T}
    d::Parameter{U}
    PolynomialClass(a::Variable{T}, c::Variable{T}, d::Variable{U}) = new(
        Parameter(a, LowerBound(zero(T), :strict)),
        Parameter(c, LowerBound(zero(T), :nonstrict)),
        Parameter(d, LowerBound(one(U),  :nonstrict))
    )
end
@outer_constructor(PolynomialClass, (1,0,3))
@inline iscomposable(::PolynomialClass, κ::Kernel) = ismercer(κ)
@inline phi{T<:AbstractFloat}(ϕ::PolynomialClass{T}, z::T) = (ϕ.a*z + ϕ.c)^ϕ.d


#== Non-Negative Negative Definite Kernel Classes ==#

abstract NonNegativeNegativeDefiniteClass{T<:AbstractFloat} <: CompositionClass{T}

doc"PowerClass(z;a,c,γ) = (az + c)ᵞ"
immutable PowerClass{T<:AbstractFloat} <: NonNegativeNegativeDefiniteClass{T}
    a::Parameter{T}
    c::Parameter{T}
    gamma::Parameter{T}
    PowerClass(a::Variable{T}, c::Variable{T}, γ::Variable{T}) = new(
        Parameter(a, LowerBound(zero(T), :strict)),
        Parameter(c, LowerBound(zero(T), :nonstrict)),
        Parameter(γ, Interval(Bound(zero(T), :strict), Bound(one(T), :nonstrict)))
    )
end
@outer_constructor(PowerClass, (1,0,0.5))
@inline iscomposable(::PowerClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::PowerClass{T}, z::T) = (ϕ.a*z + ϕ.c)^(ϕ.gamma)


doc"GammmaLogClass(z;α,γ) = log(1 + α⋅zᵞ)"
immutable GammaLogClass{T<:AbstractFloat} <: NonNegativeNegativeDefiniteClass{T}
    alpha::Parameter{T}
    gamma::Parameter{T}
    GammaLogClass(α::Variable{T}, γ::Variable{T}) = new(
        Parameter(α, LowerBound(zero(T), :strict)),
        Parameter(γ, Interval(Bound(zero(T), :strict), Bound(one(T), :nonstrict)))
    )
end
@outer_constructor(GammaLogClass, (1,0.5))
@inline iscomposable(::GammaLogClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::GammaLogClass{T}, z::T) = log(ϕ.alpha*z^(ϕ.gamma) + 1)


doc"LogClass(z;α) = log(1 + α⋅z)"
immutable LogClass{T<:AbstractFloat} <: NonNegativeNegativeDefiniteClass{T}
    alpha::Parameter{T}
    LogClass(α::Variable{T}) = new(
        Parameter(α, LowerBound(zero(T), :strict))
    )
end
@outer_constructor(LogClass, (1,))
@inline iscomposable(::LogClass, κ::Kernel) = isnegdef(κ) && isnonnegative(κ)
@inline phi{T<:AbstractFloat}(ϕ::LogClass{T}, z::T) = log(ϕ.alpha*z + 1)


#== Non-Mercer, Non-Negative Definite Classes ==#

doc"SigmoidClass(κ;α,c) = tanh(a⋅κ + c)"
immutable SigmoidClass{T<:AbstractFloat} <: CompositionClass{T}
    a::Parameter{T}
    c::Parameter{T}
    SigmoidClass(a::Variable{T}, c::Variable{T}) = new(
        Parameter(a, LowerBound(zero(T), :strict)),
        Parameter(c, LowerBound(zero(T), :nonstrict))   
    )
end
@outer_constructor(SigmoidClass, (1,0))
@inline iscomposable(::SigmoidClass, κ::Kernel) = ismercer(κ)
@inline phi{T<:AbstractFloat}(ϕ::SigmoidClass{T}, z::T) = tanh(ϕ.a*z + ϕ.c)


#== Properties of Kernel Classes ==#

for (classobj, properties) in (
        (PositiveMercerClass, (true, false, false, false, true)),
        (NonNegativeNegativeDefiniteClass, (false, true, false, true, true)),
        (PolynomialClass, (true, false, true, true, true)),
        (SigmoidClass, (false, false, true, true, true))
    )
    ismercer(::classobj) = properties[1]
    isnegdef(::classobj) = properties[2]
    attainsnegative(::classobj) = properties[3]
    attainszero(::classobj)     = properties[4]
    attainspositive(::classobj) = properties[5]
end
