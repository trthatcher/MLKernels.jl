#===================================================================================================
  Squared Distance Kernel Definitions: z = ϵᵀϵ
===================================================================================================#

#== Gamma Exponential Kernel ===============#

abstract ExponentialKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}

immutable SquaredExponentialKernel{T<:FloatingPoint} <: ExponentialKernel{T}
    alpha::T
    function SquaredExponentialKernel(α::T)
        α > 0 || throw(ArgumentError("α = $(α) must be in the range (0,∞)."))
        new(α)
    end
end
SquaredExponentialKernel{T<:FloatingPoint}(α::T = 1.0) = SquaredExponentialKernel{T}(α)

function convert{T<:FloatingPoint}(::Type{SquaredExponentialKernel{T}}, κ::SquaredExponentialKernel)
    SquaredExponentialKernel(convert(T, κ.alpha))
end

kappa{T<:FloatingPoint}(κ::SquaredExponentialKernel{T}, z::T) = exp(-κ.alpha * z)
kappa_dz{T<:FloatingPoint}(κ::SquaredExponentialKernel{T}, z::T) = -κ.alpha * exp(-κ.alpha * z)
kappa_dz2{T<:FloatingPoint}(κ::SquaredExponentialKernel{T}, z::T) = (κ.alpha^2) * exp(-κ.alpha * z)
kappa_dalpha{T<:FloatingPoint}(κ::SquaredExponentialKernel{T}, z::T) = -z * exp(-κ.alpha * z)

kappa_dp{T<:FloatingPoint}(κ::SquaredExponentialKernel{T}, param::Symbol, z::T) = param == :alpha ? kappa_dalpha(κ, z) : zero(T)

function description_string{T<:FloatingPoint}(κ::SquaredExponentialKernel{T}, eltype::Bool = true)
    "SquaredExponentialKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha))"
end

immutable GammaExponentialKernel{T<:FloatingPoint} <: ExponentialKernel{T}
    alpha::T
    gamma::T
    function GammaExponentialKernel(α::T, γ::T)
        α > 0 || throw(ArgumentError("α = $(α) must be in the range (0,∞)."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the range (0,1]."))
        new(α, γ)
    end
end
GammaExponentialKernel{T<:FloatingPoint}(α::T = 1.0, gamma::T = convert(T,0.5)) = GammaExponentialKernel{T}(α, gamma)

function convert{T<:FloatingPoint}(::Type{GammaExponentialKernel{T}}, κ::GammaExponentialKernel)
    GammaExponentialKernel(convert(T, κ.alpha), convert(T, κ.gamma))
end

kappa{T<:FloatingPoint}(κ::GammaExponentialKernel{T}, z::T) = exp(-κ.alpha * z^κ.gamma)
kappa_dz{T<:FloatingPoint}(κ::GammaExponentialKernel{T}, z::T) = -κ.alpha* κ.gamma * z^(κ.gamma - 1) * exp(-κ.alpha * z^κ.gamma)
function kappa_dz2{T<:FloatingPoint}(κ::GammaExponentialKernel{T}, z::T)
    αγ = κ.alpha * κ.gamma
    αz_γ = κ.alpha*z^(κ.gamma)
    return αγ*(z^(κ.gamma-2))*exp(-αz_γ)*(κ.gamma*αz_γ - κ.gamma + 1)
end
function kappa_dalpha{T<:FloatingPoint}(κ::GammaExponentialKernel{T}, z::T)
    z_γ = z^κ.gamma
    return -z_γ*exp(-κ.alpha * z_γ)
end
function kappa_dgamma{T<:FloatingPoint}(κ::GammaExponentialKernel{T}, z::T)
    neg_αz_γ = -κ.alpha * z^κ.gamma
    return neg_αz_γ * exp(neg_αz_γ) * log(z)
end
function kappa_dp{T<:FloatingPoint}(κ::GammaExponentialKernel{T}, param::Symbol, z::T)
    if param == :alpha
        return kappa_dalpha(κ, z)
    elseif param == :gamma
        return kappa_dgamma(κ, z)
    else
        return zero(T)
    end
end

function description_string{T<:FloatingPoint}(κ::GammaExponentialKernel{T}, eltype::Bool = true)
    "GammaExponentialKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),γ=$(κ.gamma))"
end

function convert{T<:FloatingPoint}(::Type{ExponentialKernel{T}}, κ::GammaExponentialKernel)
    GammaExponentialKernel(convert(T, κ.alpha), convert(T, κ.gamma))
end

function convert{T<:FloatingPoint}(::Type{ExponentialKernel{T}}, κ::SquaredExponentialKernel)
    SquaredExponentialKernel(convert(T, κ.alpha))
end

isposdef(::ExponentialKernel) = true

function description_string_long(::ExponentialKernel)
    """
    Gamma-Exponential Kernel:
    
    The gamma-exponential kernel is a positive definite kernel defined as:
    
        k(x,y) = exp(-α‖x-y‖²ᵞ)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, γ ∈ (0,1]
    
    Since the value of the function decreases as x and y differ, it can
    be interpreted as a similarity measure. When γ = 0.5, it is known
    as the Laplacian or exponential kernel. When γ = 1, it is known
    as the Gaussian or squared exponential kernel.

    ---
    Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian 
    Processes for Machine Learning (Adaptive Computation and Machine 
    Learning). The MIT Press.
    """
end


#== Rational Quadratic Kernel ===============#

abstract QuadraticKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}

immutable InverseQuadraticKernel{T<:FloatingPoint} <: QuadraticKernel{T}
    alpha::T
    function InverseQuadraticKernel(α::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        new(α)
    end
end
InverseQuadraticKernel{T<:FloatingPoint}(α::T = 1.0) = InverseQuadraticKernel{T}(α)

function convert{T<:FloatingPoint}(::Type{InverseQuadraticKernel{T}}, κ::InverseQuadraticKernel)
    InverseQuadraticKernel(convert(T, κ.alpha))
end

kappa{T<:FloatingPoint}(κ::InverseQuadraticKernel{T}, z::T) = 1/(1 + κ.alpha*z)
kappa_dz{T<:FloatingPoint}(κ::InverseQuadraticKernel{T}, z::T) = -κ.alpha/(1 + κ.alpha*z)^2
kappa_dz2{T<:FloatingPoint}(κ::InverseQuadraticKernel{T}, z::T) = 2κ.alpha^2/(1 + κ.alpha*z)^3
kappa_dalpha{T<:FloatingPoint}(κ::InverseQuadraticKernel{T}, z::T) = -z/(1 + κ.alpha*z)^2
kappa_dp{T<:FloatingPoint}(κ::InverseQuadraticKernel{T}, param::Symbol, z::T) = param == :alpha ? kappa_dalpha(κ, z) : zero(T)

function description_string{T<:FloatingPoint}(κ::InverseQuadraticKernel{T}, eltype::Bool = true)
    "InverseQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha))"
end

immutable RationalQuadraticKernel{T<:FloatingPoint} <: QuadraticKernel{T}
    alpha::T
    beta::T
    function RationalQuadraticKernel(α::T, β::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        β > 0 || throw(ArgumentError("β = $(β) must be greater than zero."))
        new(α, β)
    end
end
RationalQuadraticKernel{T<:FloatingPoint}(α::T = 1.0, β::T = one(T)) = RationalQuadraticKernel{T}(α, β)

function convert{T<:FloatingPoint}(::Type{RationalQuadraticKernel{T}}, κ::RationalQuadraticKernel)
    RationalQuadraticKernel(convert(T, κ.alpha), convert(T, κ.beta))
end

kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = (1 + κ.alpha*z)^(-κ.beta)
kappa_dz{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = -κ.alpha * κ.beta * (1 + κ.alpha*z)^(-κ.beta - 1)
function kappa_dz2{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T)
    α2 = κ.alpha^2
    αz = κ.alpha * z
    β = κ.beta
    return α2*β*(β + 1)*(αz + 1)^(-β-2)
end
kappa_dalpha{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = -κ.beta * z * (1 + κ.alpha*z)^(-κ.beta - 1)
kappa_dbeta{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = -log(1 + κ.alpha*z) * kappa(κ, z)

function kappa_dp{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, param::Symbol, z::T)
    if param == :alpha
        return kappa_dalpha(κ, z)
    elseif param == :beta
        return kappa_dbeta(κ, z)
    else
        return zero(T)
    end
end

function description_string{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, eltype::Bool = true)
    "RationalQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),β=$(κ.beta))"
end

function convert{T<:FloatingPoint}(::Type{QuadraticKernel{T}}, κ::InverseQuadraticKernel)
    InverseQuadraticKernel(convert(T, κ.alpha))
end

function convert{T<:FloatingPoint}(::Type{QuadraticKernel{T}}, κ::RationalQuadraticKernel)
    RationalQuadraticKernel(convert(T, κ.alpha), convert(T, κ.beta))
end

isposdef(::QuadraticKernel) = true

function description_string_long(::QuadraticKernel)
    """
    Rational Quadratic Kernel:
    
    The rational quadratic kernel is a stationary kernel that is
    similar in shape to the Gaussian kernel:
    
        k(x,y) = (1 + α‖x-y‖²/β)⁻ᵝ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, β > 0
    
    ---
    Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian 
    Processes for Machine Learning (Adaptive Computation and Machine 
    Learning). The MIT Press.
    """
end


#== Power Kernel ===============#

immutable PowerKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    gamma::T
    function PowerKernel(gamma::T)
        0 < gamma <= 1 || throw(ArgumentError("γ = $(gamma) must be in the interval (0,1]."))
        new(gamma)
    end
end
PowerKernel{T<:FloatingPoint}(gamma::T = 1.0) = PowerKernel{T}(gamma)

convert{T<:FloatingPoint}(::Type{PowerKernel{T}}, κ::PowerKernel) = PowerKernel(convert(T, κ.gamma))

kappa{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = -z^(κ.gamma)
kappa_dz{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = (-κ.gamma)*(z^(κ.gamma - 1))
kappa_dz2{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = (κ.gamma - κ.gamma^2)*(z^(κ.gamma - 2))
kappa_dgamma{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = -log(z)*(z^(κ.gamma))

function kappa_dp{T<:FloatingPoint}(κ::PowerKernel{T}, param::Symbol, z::T)
    param == :gamma ? kappa_dgamma(κ, z) : zero(T)
end

iscondposdef(::PowerKernel) = true

function description_string{T<:FloatingPoint}(κ::PowerKernel{T}, eltype::Bool = true)
    "PowerKernel" * (eltype ? "{$(T)}" : "") * "(γ=$(κ.gamma))"
end

function description_string_long(::PowerKernel)
    """
    Power Kernel:
    
    The power kernel (also known as the unrectified triangular kernel)
    is a positive semidefinite kernel. An important feature of the
    power kernel is that it is scale invariant. The function is given
    by:
    
        k(x,y) = -‖x-y‖²ᵞ   x ∈ ℝⁿ, y ∈ ℝⁿ, γ ∈ (0,1]
    
    ---
    Boughorbel, S.; Tarel, J.-P.; Nozha Boujemaa, "Conditionally 
    Positive Definite Kernels for SVM Based Image Recognition," 
    Multimedia and Expo, 2005. ICME 2005. IEEE International Conference 
    on , vol., no., pp.113,116, 6-6 July 2005
    """
end


#== Log Kernel ===============#

immutable LogKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    d::T
    function LogKernel(d::T)
        d > 0 || throw(ArgumentError("d = $(d) must be greater than zero."))
        new(d)
    end
end
LogKernel{T<:FloatingPoint}(d::T = 1.0) = LogKernel{T}(d)
LogKernel(d::Integer) = LogKernel(convert(Float64, d))

convert{T<:FloatingPoint}(::Type{LogKernel{T}}, κ::LogKernel) = LogKernel(convert(T, κ.d))

kappa{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = -log(z^(κ.d/2) + 1)
kappa_dz{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = -κ.d/(2(z^(-κ.d/2 + 1) + z))
kappa_dz2{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = κ.d/2 * (1 + z^(-κ.d/2)*(-κ.d/2 + 1))/((z^(-κ.d/2 + 1) + z)^2)
kappa_dd{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = -z^(κ.d/2)*log(z)/(2(z^(κ.d/2) + 1))

function kappa_dp{T<:FloatingPoint}(κ::LogKernel{T}, param::Symbol, z::T)
    param == :d ? kappa_dd(κ, z) : zero(T)
end

iscondposdef(::LogKernel) = true

function description_string{T<:FloatingPoint}(κ::LogKernel{T}, eltype::Bool = true)
    "LogKernel" * (eltype ? "{$(T)}" : "") * "(d=$(κ.d))"
end

function description_string_long(::LogKernel)
    """
    Log Kernel:
    
    The log kernel is a positive semidefinite kernel. The function is
    given by:
    
        k(x,y) = -log(‖x-y‖²ᵞ + 1)    x ∈ ℝⁿ, y ∈ ℝⁿ, γ ∈ (0,1]

    ---
    Boughorbel, S.; Tarel, J.-P.; Nozha Boujemaa, "Conditionally 
    Positive Definite Kernels for SVM Based Image Recognition," 
    Multimedia and Expo, 2005. ICME 2005. IEEE International Conference 
    on , vol., no., pp.113,116, 6-6 July 2005
    """
end


#== Periodic Kernel ===============#

immutable PeriodicKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    p::T
    ell::T
    function PeriodicKernel(p::T, ell::T)
        p > 0 || throw(ArgumentError("p = $(p) must be greater than zero."))
        ell > 0 || throw(ArgumentError("ell = $(ell) must be greater than zero."))
        new(p, ell)
    end
end
PeriodicKernel{T<:FloatingPoint}(p::T = 1.0, ell::T = 1.0) = PeriodicKernel{T}(p, ell)

convert{T<:FloatingPoint}(::Type{PeriodicKernel{T}}, κ::PeriodicKernel) = PeriodicKernel(convert(T, κ.p), convert(T, κ.ell))

kappa{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = exp(-2sin(π*z/κ.p)^2 / κ.ell^2)
kappa_dz{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = -2sin(2π*z/κ.p) * π/κ.p / κ.ell^2 * kappa(κ, z)
#kappa_dz2{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) =  -2π/κ.p / κ.ell^2 * (cos(2π*z/κ.p)*(2π/κ.p) * kappa(κ, z) + sin(2π*z/κ.p) * kappa_dz(κ, z))
kappa_dz2{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) =  -(2π/(κ.p * κ.ell))^2 * (cos(2π*z/κ.p) - (sin(2π*z/κ.p) / κ.ell)^2) * kappa(κ, z)
kappa_dp{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = 2sin(2π*z/κ.p) / κ.ell^2 * π*z/κ.p^2 * kappa(κ, z)
kappa_dell{T<:FloatingPoint}(κ::PeriodicKernel{T}, z::T) = 4sin(π*z/κ.p)^2 / κ.ell^3 * kappa(κ, z)

function kappa_dp{T<:FloatingPoint}(κ::PeriodicKernel{T}, param::Symbol, z::T)
    param == :p   ? kappa_dp(κ, z)   :
    param == :ell ? kappa_dell(κ, z) :
                    zero(T)
end

function description_string{T<:FloatingPoint}(κ::PeriodicKernel{T}, eltype::Bool = true)
    "PeriodicKernel" * (eltype ? "{$(T)}" : "") * "(p=$(κ.p),l=$(κ.ell))"
end

function description_string_long(::PeriodicKernel)
    """
    Periodic Kernel:
    """
end


#==========================================================================
  Conversions
==========================================================================#

for kernelobject in (:SquaredExponentialKernel,
                     :GammaExponentialKernel,
                     :InverseQuadraticKernel,
                     :RationalQuadraticKernel,
                     :PowerKernel, 
                     :LogKernel,
                     :PeriodicKernel)
    for kerneltype in (:SquaredDistanceKernel, :StandardKernel, :SimpleKernel, :Kernel)
        @eval begin
            function convert{T<:FloatingPoint}(::Type{$kerneltype{T}}, κ::$kernelobject)
                convert($kernelobject{T}, κ)
            end
        end
    end
end
