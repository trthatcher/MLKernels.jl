#==========================================================================
  Composition Classes
==========================================================================#

function is_nonneg_and_negdef(κ::Kernel)
    isnegdef(κ)      || error("Composed class must be negative definite.")
    isnonnegative(κ) || error("Composed class must attain only non-negative values.")
end

abstract PositiveMercerClass{T<:AbstractFloat} <: CompositionClass{T}

ismercer(::PositiveMercerClass) = true
attainszero(::PositiveMercerClass) = false
attainsnegative(::PositiveMercerClass) = false

function promote_arguments(θ::Variable{Real}...)
    T = eltype(θ[1])
    if (n = length(θ)) > 1
        for i = 2:n
            T = promote_type(T, eltype(θ[i]))
        end
    end
    T = T <: AbstractFloat ? T : Float64
    tuple(Variable{T}[isa(x, Fixed) ? convert(Fixed{T}, x) : convert(T, x) for x in θ]...)
end


#==========================================================================
  Exponential Class
==========================================================================#

doc"ExponentialClass(κ;α,γ) = exp(-α⋅κᵞ)"
immutable ExponentialClass{T<:AbstractFloat} <: CompositionClass{T}
    alpha::Parameter{T}
    ExponentialClass(α::Variable{T}, γ::Variable{T}) = new(Parameter(α, ℝ(:>, zero(T))))
end
ExponentialClass{T<:AbstractFloat}(α::Variable{T}) = ExponentialClass{T}(α)
ExponentialClass(α::Variable{Real}=1) = ExponentialClass(promote_arguments(α)...)

iscomposable(::ExponentialClass, κ::Kernel) = is_nonneg_and_negdef(κ)

function description_string{T<:AbstractFloat}(ϕ::ExponentialClass{T}, eltype::Bool = true)
    "Exponential" * (eltype ? "{$(T)}" : "") * "(α=$(ϕ.alpha.value))"
end

@inline phi{T<:AbstractFloat}(ϕ::ExponentialClass{T}, z::T) = exp(-ϕ.alpha * z)


#==========================================================================
  Gamma Exponential Class
==========================================================================#

doc"GammaExponentialClass(κ;α,γ) = exp(-α⋅κᵞ)"
immutable GammaExponentialClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::Parameter{T}
    gamma::Parameter{T}
    GammaExponentialClass(α::Variable{T}, γ::Variable{T}) = new(
        Parameter(α, ℝ(:>, zero(T))),
        Parameter(γ, ℝ(:>, zero(T), :(<=), one(T)))
    )
end
GammaExponentialClass{T<:AbstractFloat}(α::Variable{T}, γ::Variable{T}) = ExponentialClass{T}(α, γ)
function GammaExponentialClass(α::Variable{Real}=1, γ::Variable{Real}=1)
    GammaExponentialClass(promote_arguments(α, γ)...)
end

iscomposable(::ExponentialClass, κ::Kernel) = is_nonneg_and_negdef(κ)

function description_string{T<:AbstractFloat}(ϕ::GammaExponentialClass{T}, eltype::Bool = true)
    "GammaExponential" * (eltype ? "{$(T)}" : "") * "(α=$(ϕ.alpha.value),γ=$(ϕ.gamma.value))"
end

@inline phi{T<:AbstractFloat}(ϕ::GammaExponentialClass{T}, z::T) = exp(-ϕ.alpha * z^ϕ.gamma)


#==========================================================================
  Rational Quadratic Class
==========================================================================#

doc"RationalClass(κ;α,β,γ) = (1 + α⋅κᵞ)⁻ᵝ"
immutable RationalClass{T<:AbstractFloat} <: PositiveMercerClass{T}
    alpha::Parameter{T}
    beta::Parameter{T}
    gamma::Parameter{T}
    RationalClass(α::Variable{T}, β::Variable{T}, γ::Variable{T}) = new(
        Parameter(α, ℝ(:>, zero(T))),
        Parameter(β, ℝ(:>, zero(T))),
        Parameter(γ, ℝ(:>, zero(T), :(<=), one(T)))
    )
end
function RationalClass{T<:AbstractFloat}(α::Variable{T}, β::Variable{T}, γ::Variable{T})
    RationalClass{T}(α, β, γ)
end
function RationalClass(α::Variable{Real}=1, β::Variable{Real}=1, γ::Variable{Real}=1)
    RationalClass(promote_arguments(α, β, γ)...)
end

iscomposable(::RationalQuadraticClass, κ::Kernel) = is_nonneg_and_negdef(κ)

function description_string{T<:AbstractFloat}(ϕ::RationalQuadraticClass{T}, eltype::Bool = true)
    "GammaRational" * (eltype ? "{$(T)}" : "") * "(α=$(ϕ.alpha),β=$(ϕ.beta),γ=$(ϕ.gamma))"
end

@inline phi{T<:AbstractFloat}(ϕ::RationalQuadraticClass{T}, z::T) = (1 + ϕ.α*z^ϕ.γ)^(-ϕ.β)


#==========================================================================
  Matern Class
==========================================================================#

doc"MatérnClass(κ;ν,ρ) = 2ᵛ⁻¹(√(2ν)κ/ρ)ᵛKᵥ(√(2ν)κ/ρ)/Γ(ν)"
immutable MaternClass{T<:AbstractFloat} <: CompositionClass{T}
    nu::Parameter{T}
    rho::Parameter{T}
    MaternClass(ν::T, ν_fixed::Bool, ρ::T, ρ_fixed::Bool) = new(
        Parameter(:ν, ν_fixed, ν, ℝ(:>, zero(T))),
        Parameter(:ρ, ρ_fixed, ρ, ℝ(:>, zero(T)))
    )
end
MaternClass{T<:AbstractFloat}(ν::T, ρ::T) = MaternClass{T}(ν, false, ρ, false)
MaternClass(ν::Real=1, ρ::Real=1) = MaternClass(promote_arguments(ν, ρ)...)

iscomposable(::MaternClass, κ::Kernel) = is_nonneg_and_negdef(κ)
ismercer(::MaternClass) = true
attainszero(::MaternClass) = false
attainsnegative(::MaternClass) = false

function description_string{T<:AbstractFloat}(ϕ::MaternClass{T}, eltype::Bool = true)
    "Matérn" * (eltype ? "{$(T)}" : "") * "(" * valstring(ϕ.ν) * "," * valstring(ϕ.ρ) * ")"
end

@inline function phi{T<:AbstractFloat}(ϕ::MaternClass{T}, z::T)
    v1 = sqrt(2ϕ.nu) * z / ϕ.rho
    v1 = v1 < eps(T) ? eps(T) : v1  # Overflow risk, z -> Inf
    2 * (v1/2)^(ϕ.nu) * besselk(ϕ.nu, v1) / gamma(ϕ.nu)
end


#==========================================================================
  Polynomial Class
==========================================================================#

doc"PolynomialClass(κ;a,c,d) = (a⋅κ + c)ᵈ"
immutable PolynomialClass{T<:AbstractFloat} <: CompositionClass{T}
    a::Parameter{T}
    c::Parameter{T}
    d::Parameter
    PolynomialClass(a::T, a_fixed::Bool, c::T, c_fixed::Bool, d::Integer) = new(
        Parameter(:a, a_fixed, a, ℝ(:>, zero(T))),
        Parameter(:c, c_fixed, c, ℝ(:>, zero(T))),
        Parameter(:d, true, d, ℤ(:(>=), one(d)))
    )
end
PolynomialClass{T<:AbstractFloat}(a::T, b::T, d::Integer) = PolynomialClass{T}(a, false, b, false, d)
PolynomialClass(a::Real=1, b::Real=1, d::Integer=3) = PolynomialClass(promote_arguments(a, b)..., d)

function iscomposable(::PolynomialClass, κ::Kernel)
    ismercer(κ) || error("Composed class must be a Mercer class.")
end

ismercer(::PolynomialClass) = true

function description_string{T<:AbstractFloat}(ϕ::PolynomialClass{T}, eltype::Bool = true) 
    "Polynomial" * (eltype ? "{$(T)}" : "") * "(" * valstring(ϕ.a) * "," * valstring(ϕ.c) * "," * 
    valstring(ϕ.d) * ")"
end

@inline phi{T<:AbstractFloat}(ϕ::PolynomialClass{T}, z::T) = (ϕ.a*z + ϕ.c)^ϕ.d
#@inline phi{T<:AbstractFloat}(ϕ::PolynomialClass{T,:d1}, z::T) = ϕ.a*z + ϕ.c


#==========================================================================
  Exponentiated Class
==========================================================================#

doc"ExponentiatedClass(κ;α) = exp(a⋅κ + c)"
immutable ExponentiatedClass{T<:AbstractFloat} <: CompositionClass{T}
    a::T
    c::T
    function ExponentiatedClass(a::T, c::T)
        @assert_locationscale_ok
        new(a, c)
    end
end
function ExponentiatedClass{T<:Real}(a::T = 1.0, c::Real = zero(T))
    U = promote_type(T, typeof(c))
    U = U <: AbstractFloat ? U : Float64
    ExponentiatedClass{U}(convert(U, a), convert(U, c))
end

function iscomposable(::ExponentiatedClass, κ::Kernel)
    ismercer(κ) || error("Composed kernel must be a Mercer class.")
end

ismercer(::ExponentiatedClass) = true

attainszero(::ExponentiatedClass) = false
attainsnegative(::ExponentiatedClass) = false

function description_string{T<:AbstractFloat}(ϕ::ExponentiatedClass{T}, eltype::Bool = true)
    "Exponentiated" * (eltype ? "{$(T)}" : "") * "(a=$(ϕ.a),c=$(ϕ.c))"
end

@inline phi{T<:AbstractFloat}(ϕ::ExponentiatedClass{T}, z::T) = exp(ϕ.a*z + ϕ.c)


#==========================================================================
  Sigmoid Class
==========================================================================#

doc"SigmoidClass(κ;α,c) = tanh(a⋅κ + c)"
immutable SigmoidClass{T<:AbstractFloat} <: CompositionClass{T}
    a::T
    c::T
    function SigmoidClass(a::T, c::T)
        @assert_locationscale_ok
        new(a, c)
    end
end
function SigmoidClass{T<:Real}(a::T = 1.0, c::Real = one(T))
    U = promote_type(T, typeof(c))
    U = U <: AbstractFloat ? U : Float64
    SigmoidClass{U}(convert(U, a), convert(U, c))
end

function iscomposable(::SigmoidClass, κ::Kernel)
    ismercer(κ) || error("Composed class must be a Mercer class.")
end

function description_string{T<:AbstractFloat}(ϕ::SigmoidClass{T}, eltype::Bool = true)
    "Sigmoid" * (eltype ? "{$(T)}" : "") * "(a=$(ϕ.a),c=$(ϕ.c))"
end

@inline phi{T<:AbstractFloat}(ϕ::SigmoidClass{T}, z::T) = tanh(ϕ.a*z + ϕ.c)


#==========================================================================
  Power Class
==========================================================================#

doc"PowerClass(z;γ) = (αz + c)ᵞ"
immutable PowerClass{T<:AbstractFloat,CASE} <: CompositionClass{T}
    a::T
    c::T
    gamma::T
    function PowerClass(a::T, c::T, γ::T)
        @assert_locationscale_ok
        0 < γ <= 1 || error("γ = $(γ) must be in the interval (0,1].")
        if CASE == :γ1 && γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(a, c, γ)
    end
end
function PowerClass{T<:Real}(a::T = 1.0, c::Real = zero(T), γ::Real = one(T)/2)
    U = promote_type(T, typeof(c), typeof(γ))
    U = U <: AbstractFloat ? U : Float64
    PowerClass{U, γ == 1 ? :γ1 : :Ø}(convert(U, a), convert(U, c), convert(U, γ))
end

iscomposable(::PowerClass, κ::Kernel) = is_nonneg_and_negdef(κ)

isnegdef(::PowerClass) = true

attainszero(::PowerClass) = true
attainsnegative(::PowerClass) = false

function description_string{T<:AbstractFloat}(ϕ::PowerClass{T}, eltype::Bool = true)
    "Power" * (eltype ? "{$(T)}" : "") * "(a=$(ϕ.a),c=$(ϕ.c),γ=$(ϕ.gamma))"
end

@inline phi{T<:AbstractFloat}(ϕ::PowerClass{T}, z::T) = (ϕ.a*z + ϕ.c)^(ϕ.gamma)
@inline phi{T<:AbstractFloat}(ϕ::PowerClass{T,:γ1}, z::T) = ϕ.a*z + ϕ.c


#==========================================================================
  Log Class
==========================================================================#

doc"LogClass(z;α,γ) = log(1 + α⋅zᵞ)"
immutable LogClass{T<:AbstractFloat,CASE} <: CompositionClass{T}
    alpha::T
    gamma::T
    function LogClass(α::T, γ::T)
        α > 0 || error("α = $(α) must be greater than zero.")
        0 < γ <= 1 || error("γ = $(γ) must be in the interval (0,1].")
        if CASE == :γ1 && γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(α,γ)
    end
end
function LogClass{T<:Real}(α::T = 1.0, γ::Real = one(T))
    U = promote_type(T, typeof(γ))
    U = U <: AbstractFloat ? U : Float64
    LogClass{U, γ == 1 ? :γ1 : :Ø}(convert(U, α), convert(U, γ))
end

iscomposable(::LogClass, κ::Kernel) = is_nonneg_and_negdef(κ)

isnegdef(::LogClass) = true

attainszero(::LogClass) = true
attainsnegative(::LogClass) = false

function description_string{T<:AbstractFloat}(ϕ::LogClass{T}, eltype::Bool = true)
    "Log" * (eltype ? "{$(T)}" : "") * "(α=$(ϕ.alpha),γ=$(ϕ.gamma))"
end

@inline phi{T<:AbstractFloat}(ϕ::LogClass{T}, z::T) = log(ϕ.alpha*z^(ϕ.gamma) + 1)
@inline phi{T<:AbstractFloat}(ϕ::LogClass{T,:γ1}, z::T) = log(ϕ.alpha*z + 1)
