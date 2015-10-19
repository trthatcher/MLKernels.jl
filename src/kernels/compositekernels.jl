#==========================================================================
  Exponential Kernel
==========================================================================#

immutable ExponentialKernel{T<:AbstractFloat,CASE} <: CompositeKernel{T}
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

function ExponentialKernel{T<:AbstractFloat}(κ::BaseKernel{T}, α::T = one(T), γ::T = one(T))
    ExponentialKernel{T, γ == 1 ? :γ1 : :Ø}(κ, α, γ)
end
ExponentialKernel{T<:AbstractFloat}(α::T = 1.0, γ::T = one(T)) = ExponentialKernel(convert(Kernel{T}, SquaredDistanceKernel()), α, γ)

GaussianKernel{T<:AbstractFloat}(α::T = 1.0) = ExponentialKernel(SquaredDistanceKernel(one(T)), α)
RadialBasisKernel{T<:AbstractFloat}(α::T = 1.0) = ExponentialKernel(SquaredDistanceKernel(one(T)),α)
LaplacianKernel{T<:AbstractFloat}(α::T = 1.0) = ExponentialKernel(SquaredDistanceKernel(one(T)),α, convert(T, 0.5))

ismercer(::ExponentialKernel) = true

function description_string{T<:AbstractFloat}(κ::ExponentialKernel{T}, eltype::Bool = true)
    "ExponentialKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha),γ=$(κ.gamma))"
end

phi{T<:AbstractFloat}(κ::ExponentialKernel{T}, z::T) = exp(-κ.alpha * z^κ.gamma)
phi{T<:AbstractFloat}(κ::ExponentialKernel{T,:γ1}, z::T) = exp(-κ.alpha * z)


#==========================================================================
  Rational Quadratic Kernel
==========================================================================#

immutable RationalQuadraticKernel{T<:AbstractFloat,CASE} <: CompositeKernel{T}
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

function RationalQuadraticKernel{T<:AbstractFloat}(κ::BaseKernel{T}, α::T = one(T), β::T = one(T), γ::T = one(T))
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
function RationalQuadraticKernel{T<:AbstractFloat}(α::T = 1.0, β::T = one(T), γ::T = one(T))
    RationalQuadraticKernel(convert(Kernel{T}, SquaredDistanceKernel()), α, β, γ)
end

ismercer(::RationalQuadraticKernel) = true

function description_string{T<:AbstractFloat}(κ::RationalQuadraticKernel{T}, eltype::Bool = true)
    "RationalQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha),β=$(κ.beta),γ=$(κ.gamma))"
end

phi{T<:AbstractFloat}(κ::RationalQuadraticKernel{T}, z::T) = (1 + κ.alpha*z^κ.gamma)^(-κ.beta)
phi{T<:AbstractFloat}(κ::RationalQuadraticKernel{T,:β1γ1}, z::T) = 1/(1 + κ.alpha*z)
phi{T<:AbstractFloat}(κ::RationalQuadraticKernel{T,:β1}, z::T) = 1/(1 + κ.alpha*z^κ.gamma)
phi{T<:AbstractFloat}(κ::RationalQuadraticKernel{T,:γ1}, z::T) = (1 + κ.alpha*z)^(-κ.beta)


#==========================================================================
  Matern Kernel
==========================================================================#

immutable MaternKernel{T<:AbstractFloat,CASE} <: CompositeKernel{T}
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

MaternKernel{T<:AbstractFloat}(κ::BaseKernel{T}, ν::T = one(T), θ::T = one(T)) = MaternKernel{T, ν == 1 ? :ν1 : :Ø}(κ, ν, θ)
MaternKernel{T<:AbstractFloat}(ν::T = 1.0, θ::T = one(T)) = MaternKernel(convert(Kernel{T},SquaredDistanceKernel()), ν, θ)

ismercer(::MaternKernel) = true

function description_string{T<:AbstractFloat}(κ::MaternKernel{T}, eltype::Bool = true)
  "MatérnKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",ν=$(κ.nu),θ=$(κ.theta))"
end

function phi{T<:AbstractFloat}(κ::MaternKernel{T}, z::T)
  v1 = sqrt(2κ.nu * z)/κ.theta
  2 * (v1/2)^(κ.nu) * besselk(κ.nu, z)/gamma(κ.nu)
end

function phi{T<:AbstractFloat}(κ::MaternKernel{T,:ν1}, z::T)
  v1 = sqrt(2z)/κ.theta
  v1 * besselk(one(T), z)
end


#==========================================================================
  Power Kernel
==========================================================================#

immutable PowerKernel{T<:AbstractFloat,CASE} <: CompositeKernel{T}
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

PowerKernel{T<:AbstractFloat}(κ::BaseKernel{T}, γ::T = one(T)) = PowerKernel{T, γ == 1 ? :γ1 : :Ø}(κ, γ)
PowerKernel{T<:AbstractFloat}(γ::T = 1.0) = PowerKernel(convert(Kernel{T},SquaredDistanceKernel()), γ)

isnegdef(::PowerKernel) = true

function description_string{T<:AbstractFloat}(κ::PowerKernel{T}, eltype::Bool = true)
    "PowerKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",γ=$(κ.gamma))"
end

phi{T<:AbstractFloat}(κ::PowerKernel{T}, z::T) = z^(κ.gamma)
phi{T<:AbstractFloat}(κ::PowerKernel{T,:γ1}, z::T) = z


#==========================================================================
  Log Kernel
==========================================================================#

immutable LogKernel{T<:AbstractFloat,CASE} <: CompositeKernel{T}
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

LogKernel{T<:AbstractFloat}(κ::BaseKernel{T}, α::T = one(T), γ::T = one(T)) = LogKernel{T, γ == 1 ? :γ1 : :Ø}(κ, α, γ)
LogKernel{T<:AbstractFloat}(α::T = 1.0, γ::T = one(T)) = LogKernel(convert(Kernel{T},SquaredDistanceKernel(1.0)), α, γ)

isnegdef(::LogKernel) = true

function description_string{T<:AbstractFloat}(κ::LogKernel{T}, eltype::Bool = true)
    "LogKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha),γ=$(κ.gamma))"
end

phi{T<:AbstractFloat}(κ::LogKernel{T}, z::T) = log(κ.alpha*z^(κ.gamma) + 1)
phi{T<:AbstractFloat}(κ::LogKernel{T,:γ1}, z::T) = log(κ.alpha*z + 1)


#==========================================================================
  Polynomial Kernel
==========================================================================#

immutable PolynomialKernel{T<:AbstractFloat,CASE} <: CompositeKernel{T}
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

PolynomialKernel{T<:AbstractFloat}(κ::BaseKernel{T}, α::T = one(T), c::T = one(T), d::T = convert(T, 2)) = PolynomialKernel{T, d == 1 ? :d1 : :Ø}(κ, α, c, d)
PolynomialKernel{T<:AbstractFloat}(α::T = 1.0, c::T = one(T), d::T = convert(T, 2)) = PolynomialKernel(convert(Kernel{T},ScalarProductKernel()), α, c, d)

LinearKernel{T<:AbstractFloat}(α::T = 1.0, c::T = one(T)) = PolynomialKernel(ScalarProductKernel(), α, c, one(T))

ismercer(::PolynomialKernel) = true

function description_string{T<:AbstractFloat}(κ::PolynomialKernel{T}, eltype::Bool = true) 
    "PolynomialKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha),c=$(κ.c),d=$(convert(Int64,κ.d)))"
end

phi{T<:AbstractFloat}(κ::PolynomialKernel{T}, xᵀy::T) = (κ.alpha*xᵀy + κ.c)^κ.d
phi{T<:AbstractFloat}(κ::PolynomialKernel{T,:d1}, xᵀy::T) = κ.alpha*xᵀy + κ.c


#==========================================================================
  Exponentiated Kernel
==========================================================================#

immutable ExponentiatedKernel{T<:AbstractFloat} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    function ExponentiatedKernel(κ::BaseKernel{T}, α::T)
        ismercer(κ) == true || throw(ArgumentError("Composed kernel must be a Mercer kernel."))
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        new(κ, α)
    end
end

ExponentiatedKernel{T<:AbstractFloat}(κ::BaseKernel{T}, α::T = one(T)) = ExponentiatedKernel{T}(κ, α)
ExponentiatedKernel{T<:AbstractFloat}(α::T = 1.0) = ExponentiatedKernel(convert(Kernel{T},ScalarProductKernel()), α)

ismercer(::ExponentiatedKernel) = true

function description_string{T<:AbstractFloat}(κ::ExponentiatedKernel{T}, eltype::Bool = true)
    "ExponentiatedKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha))"
end

phi{T<:AbstractFloat}(κ::ExponentiatedKernel{T}, z::T) = exp(κ.alpha*z)


#==========================================================================
  Sigmoid Kernel
==========================================================================#

immutable SigmoidKernel{T<:AbstractFloat} <: CompositeKernel{T}
    k::BaseKernel{T}
    alpha::T
    c::T
    function SigmoidKernel(κ::BaseKernel{T}, α::T, c::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        c >= 0 || throw(ArgumentError("c = $(c) must be non-negative."))
        new(κ, α, c)
    end
end

SigmoidKernel{T<:AbstractFloat}(κ::BaseKernel{T}, α::T = one(T), c::T = one(T)) = SigmoidKernel{T}(κ, α, c)
SigmoidKernel{T<:AbstractFloat}(α::T = 1.0, c::T = one(T)) = SigmoidKernel(convert(Kernel{T},ScalarProductKernel()), α, c)

function description_string{T<:AbstractFloat}(κ::SigmoidKernel{T}, eltype::Bool = true)
    "SigmoidKernel" * (eltype ? "{$(T)}" : "") * "(κ=" * description_string(κ.k, false) * ",α=$(κ.alpha),c=$(κ.c))"
end

phi{T<:AbstractFloat}(κ::SigmoidKernel{T}, z::T) = tanh(κ.alpha*z + κ.c)
