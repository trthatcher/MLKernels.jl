#==========================================================================
  Exponential Kernel
  k(x,y) = exp(-αf(x,y)²ᵞ)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, γ ∈ (0,1]
                              f ≧ is negative definite
==========================================================================#

immutable ExponentialKernel{T<:FloatingPoint,CASE} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    gamma::T
    function ExponentialKernel(κ::BaseKernel{T}, α::T, γ::T)
        isnegdef(κ) == true || throw(ArgumentError("Composed kernel must be negative definite."))
        κ >= 0 || throw(ArgumentError("Composed kernel must attain only non-negative values."))
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))
        if CASE == :γ1 &&  γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(κ, α, γ)
    end
end
ExponentialKernel{T<:FloatingPoint}(κ::BaseKernel{T} = SquaredDistanceKernel(1.0), α::T = one(T), γ::T = one(T)) = ExponentialKernel{T, γ == 1 ? :γ1 : :Ø}(κ, α, γ)

ismercer(::ExponentialKernel) = true

function description_string{T<:FloatingPoint}(κ::ExponentialKernel{T}, eltype::Bool = true)
    "ExponentialKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha),γ=$(κ.gamma))"
end

kappa{T<:FloatingPoint}(κ::ExponentialKernel{T}, z::T) = exp(-κ.alpha * z^κ.gamma)
kappa{T<:FloatingPoint}(κ::ExponentialKernel{T,:γ1}, z::T) = exp(-κ.alpha * z)


#==========================================================================
  Rational Quadratic Kernel
  k(x,y) = (1 + α‖x-y‖²ᵞ)⁻ᵝ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, β > 0, γ ∈ (0,1]
==========================================================================#

immutable RationalQuadraticKernel{T<:FloatingPoint,CASE} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    beta::T
    gamma::T
    function RationalQuadraticKernel(κ::BaseKernel{T}, α::T, β::T, γ::T)
        isnegdef(κ) == true || throw(ArgumentError("Composed kernel must be negative definite."))
        κ >= 0 || throw(ArgumentError("Composed kernel must attain only non-negative values."))
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        β > 0 || throw(ArgumentError("β = $(β) must be greater than zero."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))      
        if CASE == :β1γ1 && (β != 1 || γ != 1)
            error("Special case β = 1 and γ = 1 flagged but β = $(β) and γ = $(γ)")
        elseif CASE == :β1 && β != 1
            error("Special case β = 1 flagged but β = $(β)")
        elseif CASE == :γ1 && γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(κ, α, β, γ)
    end
end
function RationalQuadraticKernel{T<:FloatingPoint}(κ::BaseKernel{T} = SquaredDistanceKernel(1.0), α::T = one(T), β::T = one(T), γ::T = one(T))
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
    RationalQuadraticKernel{T,CASE}(κ, α, β, γ)
end

ismercer(::RationalQuadraticKernel) = true

function description_string{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, eltype::Bool = true)
    "RationalQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha),β=$(κ.beta),γ=$(κ.gamma))"
end

kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = (1 + κ.alpha*z^κ.gamma)^(-κ.beta)
kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T,:β1γ1}, z::T) = 1/(1 + κ.alpha*z)
kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T,:β1}, z::T) = 1/(1 + κ.alpha*z^κ.gamma)
kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T,:γ1}, z::T) = (1 + κ.alpha*z)^(-κ.beta)


#==========================================================================
  Matern Kernel
  k(x,y) = ...    x ∈ ℝⁿ, y ∈ ℝⁿ, ν > 0, θ > 0
==========================================================================#

immutable MaternKernel{T<:FloatingPoint,CASE} <: CompositeKernel{T}
    k::BaseKernel{T}
    nu::T
    theta::T
    function MaternKernel(κ::BaseKernel{T}, ν::T, θ::T)
        isnegdef(κ) == true || throw(ArgumentError("Composed kernel must be negative definite."))
        κ >= 0 || throw(ArgumentError("Composed kernel must attain only non-negative values."))
        ν > 0 || throw(ArgumentError("ν = $(ν) must be greater than zero."))
        θ > 0 || throw(ArgumentError("θ = $(θ) must be greater than zero."))
        if CASE == :ν1 && ν != 1
            error("Special case ν = 1 flagged but ν = $(ν)")
        end
        new(κ, ν, θ)
    end
end
MaternKernel{T<:FloatingPoint}(κ::BaseKernel{T} = SquaredDistanceKernel(1.0), ν::T = one(T), θ::T = one(T)) = MaternKernel{T, ν == 1 ? :ν1 : :Ø}(κ, ν, θ)

ismercer(::MaternKernel) = true

function description_string{T<:FloatingPoint}(κ::MaternKernel{T}, eltype::Bool = true)
  "MaternKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",ν=$(κ.nu),θ=$(κ.theta))"
end

function kappa{T<:FloatingPoint}(κ::MaternKernel{T}, z::T)
  v1 = sqrt(2κ.nu * z)/κ.theta
  2 * (v1/2)^(κ.nu) * besselk(κ.nu, z)/gamma(κ.nu)
end

function kappa{T<:FloatingPoint}(κ::MaternKernel{T,:ν1}, z::T)
  v1 = sqrt(2z)/κ.theta
  v1 * besselk(one(T), z)
end


#==========================================================================
  Power Kernel
  k(x,y) = ‖x-y‖²ᵞ   x ∈ ℝⁿ, y ∈ ℝⁿ, γ ∈ (0,1]
==========================================================================#

immutable PowerKernel{T<:FloatingPoint,CASE} <: CompositeKernel{T}
    k::BaseKernel{T}
    gamma::T
    function PowerKernel(κ::BaseKernel{T}, γ::T)
        isnegdef(κ) == true || throw(ArgumentError("Composed kernel must be negative definite."))
        κ >= 0 || throw(ArgumentError("Composed kernel must attain only non-negative values."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))
        if CASE == :γ1 && γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(κ,γ)
    end
end
PowerKernel{T<:FloatingPoint}(κ::BaseKernel{T} = SquaredDistanceKernel(1.0), γ::T = one(T)) = PowerKernel{T, γ == 1 ? :γ1 : :Ø}(κ, γ)

isnegdef(::PowerKernel) = true

function description_string{T<:FloatingPoint}(κ::PowerKernel{T}, eltype::Bool = true)
    "PowerKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",γ=$(κ.gamma))"
end

kappa{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = z^(κ.gamma)
kappa{T<:FloatingPoint}(κ::PowerKernel{T,:γ1}, z::T) = z


#==========================================================================
  Log Kernel
  k(x,y) = log(α‖x-y‖²ᵞ + 1)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, γ ∈ (0,1]
==========================================================================#

immutable LogKernel{T<:FloatingPoint,CASE} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    gamma::T
    function LogKernel(κ::BaseKernel{T}, α::T, γ::T)
        isnegdef(κ) == true || throw(ArgumentError("Composed kernel must be negative definite."))
        κ >= 0 || throw(ArgumentError("Composed kernel must attain only non-negative values."))
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))
        if CASE == :γ1 && γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(κ,α,γ)
    end
end
LogKernel{T<:FloatingPoint}(κ::BaseKernel{T} = SquaredDistanceKernel(1.0), α::T = one(T), γ::T = one(T)) = LogKernel{T, γ == 1 ? :γ1 : :Ø}(κ, α, γ)

isnegdef(::LogKernel) = true

function description_string{T<:FloatingPoint}(κ::LogKernel{T}, eltype::Bool = true)
    "LogKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha),γ=$(κ.gamma))"
end

kappa{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = log(κ.alpha*z^(κ.gamma) + 1)
kappa{T<:FloatingPoint}(κ::LogKernel{T,:γ1}, z::T) = log(κ.alpha*z + 1)


#==========================================================================
  Polynomial Kernel
  k(x,y) = (αxᵀy + c)ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0, d > 0
==========================================================================#

immutable PolynomialKernel{T<:FloatingPoint,CASE} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    c::T
    d::T
    function PolynomialKernel(κ::BaseKernel{T}, α::T, c::T, d::T)
        ismercer(κ) == true || throw(ArgumentError("Composed kernel must be a Mercer kernel."))
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        c >= 0 || throw(ArgumentError("c = $(c) must be non-negative."))
        (d > 0 && trunc(d) == d) || throw(ArgumentError("d = $(d) must be an integer greater than zero."))
        if CASE == :d1 && d != 1
            error("Special case d = 1 flagged but d = $(convert(Int64,d))")
        end
        new(κ, α, c, d)
    end
end
PolynomialKernel{T<:FloatingPoint}(κ::BaseKernel{T}, α::T = one(T), c::T = one(T), d::T = convert(T, 2)) = PolynomialKernel{T, d == 1 ? :d1 : :Ø}(κ, α, c, d)
PolynomialKernel{T<:FloatingPoint}(κ::BaseKernel{T}, α::T, c::T, d::Integer) = PolynomialKernel(κ, α, c, convert(T, d))

ismercer(::PolynomialKernel) = true

function description_string{T<:FloatingPoint}(κ::PolynomialKernel{T}, eltype::Bool = true) 
    "PolynomialKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha),c=$(κ.c),d=$(convert(Int64,κ.d)))"
end

kappa{T<:FloatingPoint}(κ::PolynomialKernel{T}, xᵀy::T) = (κ.alpha*xᵀy + κ.c)^κ.d
kappa{T<:FloatingPoint}(κ::PolynomialKernel{T,:d1}, xᵀy::T) = κ.alpha*xᵀy + κ.c


#==========================================================================
  Exponentiated Kernel
==========================================================================#

immutable ExponentiatedKernel{T<:FloatingPoint} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    function ExponentiatedKernel(κ::BaseKernel{T}, α::T)
        ismercer(κ) == true || throw(ArgumentError("Composed kernel must be a Mercer kernel."))
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        new(κ, α)
    end
end
ExponentiatedKernel{T<:FloatingPoint}(κ::BaseKernel{T} = ScalarProductKernel(), α::T = one(T)) = ExponentiatedKernel{T}(κ, α)

ismercer(::ExponentiatedKernel) = true

function description_string{T<:FloatingPoint}(κ::ExponentiatedKernel{T}, eltype::Bool = true)
    "ExponentiatedKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha))"
end

kappa{T<:FloatingPoint}(κ::ExponentiatedKernel{T}, z::T) = exp(κ.alpha*z)


#==========================================================================
  Sigmoid Kernel
  k(x,y) = tanh(αxᵀy + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0
==========================================================================#

immutable SigmoidKernel{T<:FloatingPoint} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    c::T
    function SigmoidKernel(κ::BaseKernel{T}, α::T, c::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        c >= 0 || throw(ArgumentError("c = $(c) must be non-negative."))
        new(κ, α, c)
    end
end
SigmoidKernel{T<:FloatingPoint}(κ::BaseKernel{T} = ScalarProductKernel(), α::T = one(T), c::T = one(T)) = SigmoidKernel{T}(κ, α, c)

function description_string{T<:FloatingPoint}(κ::SigmoidKernel{T}, eltype::Bool = true)
    "SigmoidKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha),c=$(κ.c))"
end

kappa{T<:FloatingPoint}(κ::SigmoidKernel{T}, z::T) = tanh(κ.alpha*z + κ.c)
