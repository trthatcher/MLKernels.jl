#===================================================================================================
  Squared Distance Kernel Definitions: z = ϵᵀϵ
===================================================================================================#

#==========================================================================
  Exponential Kernel
==========================================================================#

abstract ExponentialKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}

isposdef(::ExponentialKernel) = true

function description_string_long(::ExponentialKernel)
    """
    Gamma-Exponential Kernel:
    
    The gamma-exponential kernel is a positive definite kernel defined as:
    
        k(x,y) = exp(-α‖x-y‖²ᵞ)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, γ ∈ (0,1]
    
    Since the value of the function decreases as x and y differ, it can
    be interpreted as a similarity measure. It is derived by exponentiating
    the conditionally positive-definite power kernel.
    
    When γ = 0.5, it is known as the Laplacian or exponential kernel. When
    γ = 1, it is known as the Gaussian or squared exponential kernel.

    ---
    Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian 
    Processes for Machine Learning (Adaptive Computation and Machine 
    Learning). The MIT Press.
    """
end

#== Squared Exponential Kernel ===============#

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

function convert{T<:FloatingPoint}(::Type{ExponentialKernel{T}}, κ::SquaredExponentialKernel)
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

#== Gamma Exponential Kernel ===============#

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

function convert{T<:FloatingPoint}(::Type{ExponentialKernel{T}}, κ::GammaExponentialKernel)
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


#==========================================================================
  Rational Kernel
==========================================================================#

abstract QuadraticKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}

isposdef(::QuadraticKernel) = true

function description_string_long(::QuadraticKernel)
    """
    Rational Kernel:
    
    The rational kernel is a stationary kernel that is similar in shape
    to the Gaussian kernel:
    
        k(x,y) = (1 + α‖x-y‖²ᵞ)⁻ᵝ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, β > 0, γ ∈ (0,1]
    
    It is derived by exponentiating the conditionally positive-definite log
    kernel. Setting α = α'/β, it can be seen that the rational kernel 
    converges to the gamma exponential kernel as β → +∞.

    When γ = 1, the kernel is referred to as the rational quadratic kernel.

    ---
    Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian 
    Processes for Machine Learning (Adaptive Computation and Machine 
    Learning). The MIT Press.
    """
end

#== Inverse Quadratic Kernel ===============#

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

function convert{T<:FloatingPoint}(::Type{QuadraticKernel{T}}, κ::InverseQuadraticKernel)
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

#== Rational Quadratic Kernel ===============#

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
    f1 = κ.alpha^2
    αz = κ.alpha * z
    β = κ.beta
    v1 = (αz + 1)^(-β-2)
    f1*β*(β + 1)*v1
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

#== Gamma Rational Kernel ===============#

immutable GammaRationalQuadraticKernel{T<:FloatingPoint} <: QuadraticKernel{T}
    alpha::T
    beta::T
    gamma::T
    function GammaRationalQuadraticKernel(α::T, β::T, γ::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        β > 0 || throw(ArgumentError("β = $(β) must be greater than zero."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the range (0,1]."))        
        new(α, β, γ)
    end
end
GammaRationalQuadraticKernel{T<:FloatingPoint}(α::T = 1.0, β::T = convert(T,2), γ::T = convert(T,0.5)) = GammaRationalQuadraticKernel{T}(α, β, γ)

function convert{T<:FloatingPoint}(::Type{GammaRationalQuadraticKernel{T}}, κ::GammaRationalQuadraticKernel)
    GammaRationalKernel(convert(T, κ.alpha), convert(T, κ.beta), convert(T, κ.gamma))
end

function convert{T<:FloatingPoint}(::Type{QuadraticKernel{T}}, κ::GammaRationalQuadraticKernel)
    GammaRationalKernel(convert(T, κ.alpha), convert(T, κ.beta), convert(T, κ.gamma))
end

kappa{T<:FloatingPoint}(κ::GammaRationalQuadraticKernel{T}, z::T) = (1 + κ.alpha*z^κ.gamma)^(-κ.beta)
function kappa_dz{T<:FloatingPoint}(κ::GammaRationalQuadraticKernel{T}, z::T)
    α = κ.alpha
    β = κ.beta
    γ = κ.gamma
    v1 = α*z^γ + 1
    f1 = -α*β*γ
    f2 = z^(γ - 1)
    f3 = v1^(-β - 1)
    f1*f2*f3
end
function kappa_dz2{T<:FloatingPoint}(κ::GammaRationalQuadraticKernel{T}, z::T)
    α = κ.alpha
    β = κ.beta
    γ = κ.gamma
    v1 = α * z^γ
    f1 = α*β*γ
    f2 = z^(γ - 2)
    f3 = (1 + v1)^(-β - 2)
    f4 = β*γ*v1 + v1 - γ + 1
    f1*f2*f3*f4
end
function kappa_dalpha{T<:FloatingPoint}(κ::GammaRationalQuadraticKernel{T}, z::T)
    f1 = z^κ.gamma
    v1 = 1 + κ.alpha*f1
    f2 = v1^(-κ.beta - 1)
    -κ.beta * f1 * f2
end
function kappa_dbeta{T<:FloatingPoint}(κ::GammaRationalQuadraticKernel{T}, z::T)
    v1 = κ.alpha * z^κ.gamma + 1
    f1 = log(v1)
    f2 = v1^(-κ.beta)
    -f1*f2
end
function kappa_dgamma{T<:FloatingPoint}(κ::GammaRationalQuadraticKernel{T}, z::T)
    f1 = κ.alpha*z^κ.gamma
    f2 = (f1 + 1)^(-κ.beta - 1)
    f3 = log(z)
    -κ.beta * f1 * f2 * f3
end

function kappa_dp{T<:FloatingPoint}(κ::GammaRationalQuadraticKernel{T}, param::Symbol, z::T)
    if param == :alpha
        return kappa_dalpha(κ, z)
    elseif param == :beta
        return kappa_dbeta(κ, z)
    elseif param == :gamma
        return kappa_dgamma(κ, z)
    else
        return zero(T)
    end
end

function description_string{T<:FloatingPoint}(κ::GammaRationalQuadraticKernel{T}, eltype::Bool = true)
    "GammaRationalQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),β=$(κ.beta),γ=$(κ.gamma))"
end


#==========================================================================
  Power Kernel
==========================================================================#

abstract PowerKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}

iscondposdef(::PowerKernel) = true

function description_string_long(::PowerKernel)
    """
    Power Kernel:
    
    The power kernel (also known as the unrectified triangular kernel)
    is a positive semidefinite kernel. An important feature of the
    power kernel is that it is scale invariant. The function is given
    by:
    
        k(x,y) = -‖x-y‖²ᵞ   x ∈ ℝⁿ, y ∈ ℝⁿ, γ ∈ (0,1]
    
    ---
    Boughorbel, S.; Tarel, J.-P.; Nozha Boujemaa, Conditionally 
    Positive Definite Kernels for SVM Based Image Recognition, 
    Multimedia and Expo, 2005. ICME 2005. IEEE International Conference 
    on , vol., no., pp.113,116, 6-6 July 2005
    """
end

#== Distance Power Kernel ===============#

# Distance Power kernel (gamma = 1) goes here.

#== Gamma Power Kernel ===============#

immutable GammaPowerKernel{T<:FloatingPoint} <: PowerKernel{T}
    gamma::T
    function GammaPowerKernel(γ::T)
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))
        new(γ)
    end
end
GammaPowerKernel{T<:FloatingPoint}(γ::T = 1.0) = GammaPowerKernel{T}(γ)

convert{T<:FloatingPoint}(::Type{GammaPowerKernel{T}}, κ::GammaPowerKernel) = GammaPowerKernel(convert(T, κ.gamma))
convert{T<:FloatingPoint}(::Type{PowerKernel{T}}, κ::GammaPowerKernel) = GammaPowerKernel(convert(T, κ.gamma))

kappa{T<:FloatingPoint}(κ::GammaPowerKernel{T}, z::T) = -z^(κ.gamma)
kappa_dz{T<:FloatingPoint}(κ::GammaPowerKernel{T}, z::T) = -κ.gamma*(z^(κ.gamma - 1))
kappa_dz2{T<:FloatingPoint}(κ::GammaPowerKernel{T}, z::T) = (κ.gamma - κ.gamma^2)*(z^(κ.gamma - 2))
kappa_dgamma{T<:FloatingPoint}(κ::GammaPowerKernel{T}, z::T) = -log(z)*(z^(κ.gamma))

kappa_dp{T<:FloatingPoint}(κ::GammaPowerKernel{T}, param::Symbol, z::T) = param == :gamma ? kappa_dgamma(κ, z) : zero(T)

function description_string{T<:FloatingPoint}(κ::GammaPowerKernel{T}, eltype::Bool = true)
    "GammaPowerKernel" * (eltype ? "{$(T)}" : "") * "(γ=$(κ.gamma))"
end

#==========================================================================
  Log Kernel
==========================================================================#

# Split off into LogDistanceKernel & GammaLogKernel <: LogKernel{T}

immutable LogKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    alpha::T
    gamma::T
    function LogKernel(α::T, γ::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        γ > 0 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))
        new(α,γ)
    end
end
LogKernel{T<:FloatingPoint}(α::T = 1.0, γ::T = convert(T,0.5)) = LogKernel{T}(α,γ)

convert{T<:FloatingPoint}(::Type{LogKernel{T}}, κ::LogKernel) = LogKernel(convert(T, κ.alpha), convert(T, κ.gamma))

kappa{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = -log(κ.alpha*z^(κ.gamma) + 1)
kappa_dz{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = - κ.alpha * κ.gamma * z^(κ.gamma-1) / (κ.alpha*z^(κ.gamma) + 1)
kappa_dz2{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = - κ.alpha * κ.gamma * z^(κ.gamma-2) * (κ.gamma - 1 - κ.alpha*z^(κ.gamma)) / (κ.alpha*z^(κ.gamma) + 1)^2
kappa_dalpha{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = - z^(κ.gamma) / (κ.alpha*z^(κ.gamma) + 1)
kappa_dgamma{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = - log(z) * κ.alpha * z^(κ.gamma) / (κ.alpha*z^(κ.gamma) + 1)

function kappa_dp{T<:FloatingPoint}(κ::LogKernel{T}, param::Symbol, z::T)
    if param == :alpha
        return kappa_dalpha(κ, z)
    elseif param == :gamma
        return kappa_dgamma(κ, z)
    else
        return zero(T)
    end
end

iscondposdef(::LogKernel) = true

function description_string{T<:FloatingPoint}(κ::LogKernel{T}, eltype::Bool = true)
    "LogKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),γ=$(κ.gamma))"
end

function description_string_long(::LogKernel)
    """
    Log Kernel:
    
    The log kernel is a positive semidefinite kernel. The function is
    given by:
    
        k(x,y) = -log(α‖x-y‖²ᵞ + 1)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, γ ∈ (0,1]

    ---
    Boughorbel, S.; Tarel, J.-P.; Nozha Boujemaa, Conditionally 
    Positive Definite Kernels for SVM Based Image Recognition,
    Multimedia and Expo, 2005. ICME 2005. IEEE International Conference 
    on , vol., no., pp.113,116, 6-6 July 2005
    """
end


#==========================================================================
  Periodic Kernel
==========================================================================#

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
