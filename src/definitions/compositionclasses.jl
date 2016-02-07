#==========================================================================
  Composition Classes
==========================================================================#

function is_nonneg_and_negdef(κ::Kernel)
    isnegdef(κ)      || error("Composed class must be negative definite.")
    isnonnegative(κ) || error("Composed class must attain only non-negative values.")
end

macro assert_locationscale_ok()
    quote
        a >  0 || error("a = $(a) must be greater than zero.")
        c >= 0 || error("c = $(c) must be non-negative.")
    end
end

function promote_arguments(θ::Real...)
    T = typeof(θ[1])
    if (n = length(θ)) > 1
        for i = 2:n
            T = promote_type(T, typeof(θ[i]))
        end
    end
    T = T <: AbstractFloat ? T : Float64
    tuple(T[θ...]...)
end


#==========================================================================
  Exponential Class
==========================================================================#

doc"ExponentialClass(κ;α,γ) = exp(-α⋅κᵞ)"
immutable ExponentialClass{T<:AbstractFloat} <: CompositionClass{T}
    α::Parameter{T}
    γ::Parameter{T}
    ExponentialClass(α::T, α_fixed::Bool, γ::T, γ_fixed::Bool) = new(
        Parameter(:α, α_fixed, α, ℝ(:>,zero(T))),
        Parameter(:γ, γ_fixed, γ, ℝ(:>, zero(T), :(<=), one(T)))
    )
end
ExponentialClass{T<:AbstractFloat}(α::T, γ::T) = ExponentialClass{T}(α, false, γ, false)
ExponentialClass(α::Real=1, γ::Real=1) = ExponentialClass(promote_arguments(α, γ)...)

iscomposable(::ExponentialClass, κ::Kernel) = is_nonneg_and_negdef(κ)
ismercer(::ExponentialClass) = true
attainszero(::ExponentialClass) = false
attainsnegative(::ExponentialClass) = false

function description_string{T<:AbstractFloat}(ϕ::ExponentialClass{T}, eltype::Bool = true)
    "Exponential" * (eltype ? "{$(T)}" : "") * "(" * valstring(ϕ.α) * "," * valstring(ϕ.γ) * ")"
end

@inline phi{T<:AbstractFloat}(ϕ::ExponentialClass{T}, z::T) = exp(-ϕ.α * z^ϕ.γ)
#@inline phi{T<:AbstractFloat}(ϕ::ExponentialClass{T,:γ1}, z::T) = exp(-ϕ.α * z)


#==========================================================================
  Rational Quadratic Class
==========================================================================#

doc"RationalQuadraticClass(κ;α,β,γ) = (1 + α⋅κᵞ)⁻ᵝ"
immutable RationalQuadraticClass{T<:AbstractFloat,CASE} <: CompositionClass{T}
    alpha::T
    beta::T
    gamma::T
    function RationalQuadraticClass(α::T, β::T, γ::T)
        α > 0 || error("α = $(α) must be greater than zero.")
        β > 0 || error("β = $(β) must be greater than zero.")
        0 < γ <= 1 || error("γ = $(γ) must be in the interval (0,1].")
        if CASE == :β1γ1 && (β != 1 || γ != 1)
            error("Special case β = 1 and γ1 flagged but β = $(β) and γ = $(γ)")
        elseif CASE == :β1 && β != 1
            error("Special case β = 1 flagged but β = $(β)")
        elseif CASE == :γ1 && γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(α, β, γ)
    end
end

function RationalQuadraticClass{T<:Real}(α::T = 1.0, β::Real = one(T), γ::Real = one(T))
    U = promote_type(T, typeof(β), typeof(γ))
    U = U <: AbstractFloat ? U : Float64
    β1 = β == 1
    γ1 = γ == 1
    CASE =  if β1 && γ1
                :β1γ1
            elseif β1
                :β1
            elseif γ1
                :γ1
            else
                :Ø
            end    
    RationalQuadraticClass{U,CASE}(convert(U, α), convert(U, β), convert(U, γ))
end

iscomposable(::RationalQuadraticClass, κ::Kernel) = is_nonneg_and_negdef(κ)

ismercer(::RationalQuadraticClass) = true

attainszero(::RationalQuadraticClass) = false
attainsnegative(::RationalQuadraticClass) = false

function description_string{T<:AbstractFloat}(ϕ::RationalQuadraticClass{T}, eltype::Bool = true)
    "RationalQuadratic" * (eltype ? "{$(T)}" : "") *"(α=$(ϕ.alpha),β=$(ϕ.beta),γ=$(ϕ.gamma))"
end

@inline function phi{T<:AbstractFloat}(ϕ::RationalQuadraticClass{T}, z::T)
    (1 + ϕ.alpha*z^ϕ.gamma)^(-ϕ.beta)
end
@inline phi{T<:AbstractFloat}(ϕ::RationalQuadraticClass{T,:β1γ1}, z::T) = 1/(1 + ϕ.alpha*z)
@inline phi{T<:AbstractFloat}(ϕ::RationalQuadraticClass{T,:β1}, z::T) = 1/(1 + ϕ.alpha*z^ϕ.gamma)
@inline phi{T<:AbstractFloat}(ϕ::RationalQuadraticClass{T,:γ1}, z::T) = (1 + ϕ.alpha*z)^(-ϕ.beta)


#==========================================================================
  Matern Class
==========================================================================#

doc"MatérnClass(κ;ν,θ) = 2ᵛ⁻¹(√(2ν)κ/θ)ᵛKᵥ(√(2ν)κ/θ)/Γ(ν)"
immutable MaternClass{T<:AbstractFloat,CASE} <: CompositionClass{T}
    nu::T
    theta::T
    function MaternClass(ν::T, θ::T)
        ν > 0 || error("ν = $(ν) must be greater than zero.")
        θ > 0 || error("θ = $(θ) must be greater than zero.")
        if CASE == :ν1 && ν != 1
            error("Special case ν = 1 flagged but ν = $(ν)")
        end
        new(ν, θ)
    end
end
function MaternClass{T<:Real}(ν::T = 1.0, θ::Real = one(T))
    U = promote_type(T, typeof(θ))
    U = U <: AbstractFloat ? U : Float64
    MaternClass{U, ν == 1 ? :ν1 : :Ø}(convert(U, ν), convert(U, θ))
end

iscomposable(::MaternClass, κ::Kernel) = is_nonneg_and_negdef(κ)

ismercer(::MaternClass) = true

attainszero(::MaternClass) = false
attainsnegative(::MaternClass) = false

function description_string{T<:AbstractFloat}(ϕ::MaternClass{T}, eltype::Bool = true)
    "Matérn" * (eltype ? "{$(T)}" : "") * "(ν=$(ϕ.nu),θ=$(ϕ.theta))"
end

@inline function phi{T<:AbstractFloat}(ϕ::MaternClass{T}, z::T)
    v1 = sqrt(2ϕ.nu) * z / ϕ.theta
    v1 = v1 < eps(T) ? eps(T) : v1  # Overflow risk, z -> Inf
    2 * (v1/2)^(ϕ.nu) * besselk(ϕ.nu, v1) / gamma(ϕ.nu)
end

@inline function phi{T<:AbstractFloat}(ϕ::MaternClass{T,:ν1}, z::T)
    v1 = sqrt(2) * z / ϕ.theta
    v1 = v1 < eps(T) ? eps(T) : v1  # Overflow risk, z -> Inf
    2 * (v1/2) * besselk(one(T), v1)
end


#==========================================================================
  Polynomial Class
==========================================================================#

doc"PolynomialClass(κ;a,c,d) = (a⋅κ + c)ᵈ"
immutable PolynomialClass{T<:AbstractFloat,CASE} <: CompositionClass{T}
    a::T
    c::T
    d::T
    function PolynomialClass(a::T, c::T, d::T)
        @assert_locationscale_ok
        (d > 0 && trunc(d) == d) || error("d = $(d) must be an integer greater than zero.")
        if CASE == :d1 && d != 1
            error("Special case d = 1 flagged but d = $(convert(Int64,d))")
        end
        new(a, c, d)
    end
end
function PolynomialClass{T<:Real}(a::T = 1.0, c::Real = one(T), d::Real = 3one(T))
    U = promote_type(T, typeof(c), typeof(d))
    U = U <: AbstractFloat ? U : Float64
    PolynomialClass{U, d == 1 ? :d1 : :Ø}(convert(U, a), convert(U, c), convert(U, d))
end

function iscomposable(::PolynomialClass, κ::Kernel)
    ismercer(κ) || error("Composed class must be a Mercer class.")
end

ismercer(::PolynomialClass) = true

function description_string{T<:AbstractFloat}(ϕ::PolynomialClass{T}, eltype::Bool = true) 
    "Polynomial" * (eltype ? "{$(T)}" : "") * "(a=$(ϕ.a),c=$(ϕ.c),d=$(convert(Int64,ϕ.d)))"
end

@inline phi{T<:AbstractFloat}(ϕ::PolynomialClass{T}, z::T) = (ϕ.a*z + ϕ.c)^ϕ.d
@inline phi{T<:AbstractFloat}(ϕ::PolynomialClass{T,:d1}, z::T) = ϕ.a*z + ϕ.c


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
