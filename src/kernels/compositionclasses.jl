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


#==========================================================================
  Exponential Class
==========================================================================#

doc"ExponentialClass(z;α,γ) = exp(-α⋅zᵞ)"
immutable ExponentialClass{T<:AbstractFloat,CASE} <: CompositionClass{T}
    alpha::T
    gamma::T
    function ExponentialClass(α::T, γ::T)
        α > 0 || error("α = $(α) must be greater than zero.")
        0 < γ <= 1 || error("γ = $(γ) must be in the interval (0,1].")
        if CASE == :γ1 &&  γ != 1 
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(α, γ)
    end
end
function ExponentialClass{T<:AbstractFloat}(α::T = 1.0, γ::T = one(T))
    ExponentialClass{T, γ == 1 ? :γ1 : :Ø}(α, γ)
end

iscomposable(::ExponentialClass, κ::Kernel) = is_nonneg_and_negdef(κ)
ismercer(::ExponentialClass) = true
kernelrange(::ExponentialClass) = :Rp
attainszero(::ExponentialClass) = false

function description_string{T<:AbstractFloat}(ϕ::ExponentialClass{T}, eltype::Bool = true)
    "Exponential" * (eltype ? "{$(T)}" : "") * "(α=$(ϕ.alpha),γ=$(ϕ.gamma))"
end

@inline phi{T<:AbstractFloat}(ϕ::ExponentialClass{T}, z::T) = exp(-ϕ.alpha * z^ϕ.gamma)
@inline phi{T<:AbstractFloat}(ϕ::ExponentialClass{T,:γ1}, z::T) = exp(-ϕ.alpha * z)


#==========================================================================
  Rational Quadratic Class
==========================================================================#

doc"RationalQuadraticClass(z;α,β,γ) = (1 + α⋅zᵞ)⁻ᵝ"
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

function RationalQuadraticClass{T<:AbstractFloat}(α::T = 1.0, β::T = one(T), γ::T = one(T))
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
    RationalQuadraticClass{T,CASE}(α, β, γ)
end

iscomposable(::RationalQuadraticClass, κ::Kernel) = is_nonneg_and_negdef(κ)
ismercer(::RationalQuadraticClass) = true
kernelrange(::RationalQuadraticClass) = :Rp
attainszero(::RationalQuadraticClass) = false

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

doc"MatérnClass(z;ν,θ) = 2ᵛ⁻¹(√(2ν)z/θ)ᵛKᵥ(√(2ν)z/θ)/Γ(ν)"
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
MaternClass{T<:AbstractFloat}(ν::T = 1.0, θ::T = one(T)) = MaternClass{T, ν == 1 ? :ν1 : :Ø}(ν, θ)

iscomposable(::MaternClass, κ::Kernel) = is_nonneg_and_negdef(κ)
ismercer(::MaternClass) = true
kernelrange(::MaternClass) = :Rp
attainszero(::MaternClass) = false

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

doc"PolynomialClass(z;a,c,d) = (a⋅z + c)ᵈ"
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
function PolynomialClass{T<:AbstractFloat}(a::T = 1.0, c::T = one(T), d::T = 3one(T))
    PolynomialClass{T, d == 1 ? :d1 : :Ø}(a, c, d)
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

doc"ExponentiatedClass(z;α) = exp(a⋅z + c)"
immutable ExponentiatedClass{T<:AbstractFloat} <: CompositionClass{T}
    a::T
    c::T
    function ExponentiatedClass(a::T, c::T)
        @assert_locationscale_ok
        new(a, c)
    end
end

ExponentiatedClass{T<:AbstractFloat}(a::T = 1.0, c::T = zero(T)) = ExponentiatedClass{T}(a, c)

function iscomposable(::ExponentiatedClass, κ::Kernel)
    ismercer(κ) || error("Composed class must be a Mercer class.")
end
ismercer(::ExponentiatedClass) = true
kernelrange(::ExponentiatedClass) = :Rp
attainszero(::ExponentiatedClass) = false

function description_string{T<:AbstractFloat}(ϕ::ExponentiatedClass{T}, eltype::Bool = true)
    "Exponentiated" * (eltype ? "{$(T)}" : "") * "(a=$(ϕ.a),c=$(ϕ.c))"
end

@inline phi{T<:AbstractFloat}(ϕ::ExponentiatedClass{T}, z::T) = exp(ϕ.a*z + ϕ.c)


#==========================================================================
  Sigmoid Class
==========================================================================#

doc"SigmoidClass(z;α,c) = tanh(a⋅z + c)"
immutable SigmoidClass{T<:AbstractFloat} <: CompositionClass{T}
    a::T
    c::T
    function SigmoidClass(a::T, c::T)
        @assert_locationscale_ok
        new(a, c)
    end
end

SigmoidClass{T<:AbstractFloat}(a::T = 1.0, c::T = one(T)) = SigmoidClass{T}(a, c)

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
function PowerClass{T<:AbstractFloat}(a::T = 1.0, c = zero(T), γ::T = one(T)/2)
    PowerClass{T, γ == 1 ? :γ1 : :Ø}(a, c, γ)
end

iscomposable(::PowerClass, κ::Kernel) = is_nonneg_and_negdef(κ)
isnegdef(::PowerClass) = true
kernelrange(::PowerClass) = :Rp
attainszero(::PowerClass) = true

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
LogClass{T<:AbstractFloat}(α::T = 1.0, γ::T = one(T)) = LogClass{T, γ == 1 ? :γ1 : :Ø}(α, γ)

iscomposable(::LogClass, κ::Kernel) = is_nonneg_and_negdef(κ)
isnegdef(::LogClass) = true
kernelrange(::LogClass) = :Rp
attainszero(::LogClass) = true

function description_string{T<:AbstractFloat}(ϕ::LogClass{T}, eltype::Bool = true)
    "Log" * (eltype ? "{$(T)}" : "") * "(α=$(ϕ.alpha),γ=$(ϕ.gamma))"
end

@inline phi{T<:AbstractFloat}(ϕ::LogClass{T}, z::T) = log(ϕ.alpha*z^(ϕ.gamma) + 1)
@inline phi{T<:AbstractFloat}(ϕ::LogClass{T,:γ1}, z::T) = log(ϕ.alpha*z + 1)
