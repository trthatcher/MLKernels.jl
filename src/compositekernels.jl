#==========================================================================
  Exponential Kernel
  k(x,y) = exp(-αf(x,y)²ᵞ)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, γ ∈ (0,1]
                              f ≧ is negative definite
==========================================================================#

immutable ExponentialKernel{T<:FloatingPoint,CASE} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    gamma::T
    function ExponentialKernel(k::BaseKernel{T}, α::T, γ::T)
        isnegdef(k) == true || throw(ArgumentError("Composed kernel must be negative definite."))
        k >= 0 || throw(ArgumentError("Composed kernel must attain only non-negative values."))
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))
        if CASE == :γ1 &&  γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(k, α, γ)
    end
end
ExponentialKernel{T<:FloatingPoint}(k::BaseKernel{T}, α::T = one(T), γ::T = one(T)) = ExponentialKernel{T, γ == 1 ? :γ1 : :Ø}(k, α, γ)

#GaussianKernel{T<:FloatingPoint}(α::T = 1.0) = ExponentialKernel(α)
#RadialBasisKernel{T<:FloatingPoint}(α::T = 1.0) = ExponentialKernel(α)
#LaplacianKernel{T<:FloatingPoint}(α::T = 1.0) = ExponentialKernel(α, convert(T, 0.5))

ismercer(::ExponentialKernel) = true

function description_string{T<:FloatingPoint}(κ::ExponentialKernel{T}, eltype::Bool = true)
    "ExponentialKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),γ=$(κ.gamma))"
end

kappa{T<:FloatingPoint}(κ::ExponentialKernel{T}, z::T) = exp(-κ.alpha * z^κ.gamma)
kappa{T<:FloatingPoint}(κ::ExponentialKernel{T,:γ1}, z::T) = exp(-κ.alpha * z)


#==========================================================================
  Rational Quadratic Kernel
  k(x,y) = (1 + α‖x-y‖²ᵞ)⁻ᵝ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, β > 0, γ ∈ (0,1]
==========================================================================#

#=

immutable RationalQuadraticKernel{T<:FloatingPoint,CASE} <: SquaredDistanceKernel{T}
    alpha::T
    beta::T
    gamma::T
    function RationalQuadraticKernel(α::T, β::T, γ::T)
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
        new(α, β, γ)
    end
end
function RationalQuadraticKernel{T<:FloatingPoint}(α::T = 1.0, β::T = one(T), γ::T = one(T))
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
    RationalQuadraticKernel{T,CASE}(α, β, γ)
end

ismercer(::RationalQuadraticKernel) = true

function description_string{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, eltype::Bool = true)
    "RationalQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),β=$(κ.beta),γ=$(κ.gamma))"
end

kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = (1 + κ.alpha*z^κ.gamma)^(-κ.beta)
kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T,:β1γ1}, z::T) = 1/(1 + κ.alpha*z)
kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T,:β1}, z::T) = 1/(1 + κ.alpha*z^κ.gamma)
kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T,:γ1}, z::T) = (1 + κ.alpha*z)^(-κ.beta)

=#

#==========================================================================
  Power Kernel
  k(x,y) = -‖x-y‖²ᵞ   x ∈ ℝⁿ, y ∈ ℝⁿ, γ ∈ (0,1]
==========================================================================#

#=
immutable PowerKernel{T<:FloatingPoint,CASE} <: SquaredDistanceKernel{T}
    gamma::T
    function PowerKernel(γ::T)
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))
        if CASE == :γ1 && γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(γ)
    end
end
PowerKernel{T<:FloatingPoint}(γ::T = 1.0) = PowerKernel{T, γ == 1 ? :γ1 : :Ø}(γ)

iscondposdef(::PowerKernel) = true

function description_string{T<:FloatingPoint}(κ::PowerKernel{T}, eltype::Bool = true)
    "PowerKernel" * (eltype ? "{$(T)}" : "") * "(γ=$(κ.gamma))"
end

kappa{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = -z^(κ.gamma)
kappa{T<:FloatingPoint}(κ::PowerKernel{T,:γ1}, z::T) = -z

=#
#==========================================================================
  Log Kernel
  k(x,y) = -log(α‖x-y‖²ᵞ + 1)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, γ ∈ (0,1]
==========================================================================#

#=
immutable LogKernel{T<:FloatingPoint,CASE} <: SquaredDistanceKernel{T}
    alpha::T
    gamma::T
    function LogKernel(α::T, γ::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))
        if CASE == :γ1 && γ != 1
            error("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(α,γ)
    end
end
LogKernel{T<:FloatingPoint}(α::T = 1.0, γ::T = one(T)) = LogKernel{T, γ == 1 ? :γ1 : :Ø}(α, γ)

iscondposdef(::LogKernel) = true

function description_string{T<:FloatingPoint}(κ::LogKernel{T}, eltype::Bool = true)
    "LogKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),γ=$(κ.gamma))"
end

kappa{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = -log(κ.alpha*z^(κ.gamma) + 1)
kappa{T<:FloatingPoint}(κ::LogKernel{T,:γ1}, z::T) = -log(κ.alpha*z + 1)
=#

#==========================================================================
  Matern Kernel
  k(x,y) = ...    x ∈ ℝⁿ, y ∈ ℝⁿ, ν > 0, θ > 0
==========================================================================#
#=
immutable MaternKernel{T<:FloatingPoint,CASE} <: SquaredDistanceKernel{T}
    nu::T
    theta::T
    function MaternKernel(ν::T, θ::T)
        ν > 0 || throw(ArgumentError("ν = $(ν) must be greater than zero."))
        θ > 0 || throw(ArgumentError("θ = $(θ) must be greater than zero."))
        if CASE == :ν1 && ν != 1
            error("Special case ν = 1 flagged but ν = $(ν)")
        end
        new(ν, θ)
    end
end
MaternKernel{T<:FloatingPoint}(ν::T = 1.0, θ::T = one(T)) = MaternKernel{T, ν == 1 ? :ν1 : :Ø}(ν, θ)

ismercer(::MaternKernel) = true

function description_string{T<:FloatingPoint}(κ::MaternKernel{T}, eltype::Bool = true)
    "MaternKernel" * (eltype ? "{$(T)}" : "") * "(ν=$(κ.nu),θ=$(κ.theta))"
end

function kappa{T<:FloatingPoint}(κ::MaternKernel{T}, z::T)
    v1 = sqrt(2κ.nu * z)/κ.theta
    2 * (v1/2)^(κ.nu) * besselk(κ.nu, z)/gamma(κ.nu)
end

function kappa{T<:FloatingPoint}(κ::MaternKernel{T,:ν1}, z::T)
    v1 = sqrt(2z)/κ.theta
    v1 * besselk(one(T), z)
end

=#
