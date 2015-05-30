#===================================================================================================
  Squared Distance Kernel Definitions: z = ϵᵀϵ
===================================================================================================#

#==========================================================================
  Exponential Kernel
==========================================================================#

immutable ExponentialKernel{T<:FloatingPoint,CASE} <: SquaredDistanceKernel{T}
    alpha::T
    gamma::T
    function ExponentialKernel(α::T, γ::T)
        α > 0 || throw(ArgumentError("α = $(α) must be in the range (0,∞)."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the range (0,1]."))
        if CASE == :γ1 &&  γ != 1
            warn("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(α, γ)
    end
end
ExponentialKernel{T<:FloatingPoint}(α::T = 1.0, γ::T = one(T)) = ExponentialKernel{T, γ == 1 ? :γ1 : :Ø}(α, γ)

ismercer(::ExponentialKernel) = true

function description_string{T<:FloatingPoint}(κ::ExponentialKernel{T}, eltype::Bool = true)
    "ExponentialKernel" * (eltype ? "{$(T)}" : "") * "(α=$(κ.alpha),γ=$(κ.gamma))"
end

function description_string_long(::ExponentialKernel)
    """
    Exponential Kernel:
    
    The exponential kernel is a positive definite kernel defined as:
    
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

kappa{T<:FloatingPoint}(κ::ExponentialKernel{T}, z::T) = exp(-κ.alpha * z^κ.gamma)
kappa{T<:FloatingPoint}(κ::ExponentialKernel{T,:γ1}, z::T) = exp(-κ.alpha * z)

kappa_dz{T<:FloatingPoint}(κ::ExponentialKernel{T}, z::T) = -κ.alpha * κ.gamma * z^(κ.gamma - 1) * exp(-κ.alpha * z^κ.gamma)
kappa_dz{T<:FloatingPoint}(κ::ExponentialKernel{T,:γ1}, z::T) = -κ.alpha * exp(-κ.alpha * z)

function kappa_dz2{T<:FloatingPoint}(κ::ExponentialKernel{T}, z::T)
    v1 = κ.alpha*z^(κ.gamma)
    κ.alpha * κ.gamma * (z^(κ.gamma-2)) * exp(-v1) * (κ.gamma*v1 - κ.gamma + 1)
end

function kappa_dalpha{T<:FloatingPoint}(κ::ExponentialKernel{T}, z::T)
    v1 = z^κ.gamma
    -v1 * exp(-κ.alpha * v1)
end

function kappa_dgamma{T<:FloatingPoint}(κ::ExponentialKernel{T}, z::T)
    v1 = -κ.alpha * z^κ.gamma
    v1 * exp(v1) * log(z)
end

function kappa_dp{T<:FloatingPoint}(κ::ExponentialKernel{T}, param::Symbol, z::T)
    if param == :alpha
        kappa_dalpha(κ, z)
    elseif param == :gamma
        kappa_dgamma(κ, z)
    else
        zero(T)
    end
end


#==========================================================================
  Rational Quadratic Kernel
==========================================================================#

immutable RationalQuadraticKernel{T<:FloatingPoint,CASE} <: SquaredDistanceKernel{T}
    alpha::T
    beta::T
    gamma::T
    function RationalQuadraticKernel(α::T, β::T, γ::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        β > 0 || throw(ArgumentError("β = $(β) must be greater than zero."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the range (0,1]."))      
        if CASE == :β1γ1 && (β != 1 || γ != 1)
            warn("Special case β = 1 and γ = 1 flagged but β = $(β) and γ = $(γ)")
        elseif CASE == :β1 && β != 1
            warn("Special case β = 1 flagged but β = $(β)")
        elseif CASE == :γ1 && γ != 1
            warn("Special case γ = 1 flagged but γ = $(γ)")
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

function description_string_long(::RationalQuadraticKernel)
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

kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = (1 + κ.alpha*z^κ.gamma)^(-κ.beta)
kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T,:β1γ1}, z::T) = 1/(1 + κ.alpha*z)
kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T,:β1}, z::T) = 1/(1 + κ.alpha*z^κ.gamma)
kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T,:γ1}, z::T) = (1 + κ.alpha*z)^(-κ.beta)

function kappa_dz{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T)
    v1 = κ.alpha * z^κ.gamma + 1
    -κ.alpha * κ.beta * κ.gamma * z^(κ.gamma - 1) * (v1^(-κ.beta - 1))
end

function kappa_dz2{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T)
    v1 = κ.alpha * z^κ.gamma
    v2 = κ.beta * κ.gamma
    κ.alpha * v2 * (z^(κ.gamma - 2)) * ((1 + v1)^(-κ.beta - 2)) * (v2*v1 + v1 - γ + 1)
end

function kappa_dalpha{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T)
    v1 = 1 + κ.alpha*f1
    v2 = -κ.beta
    v2 * z^κ.gamma * v1^(v2 - 1)
end
function kappa_dbeta{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T)
    v1 = κ.alpha * z^κ.gamma + 1
    -log(v1) * v1^(-κ.beta)
end
function kappa_dgamma{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T)
    v1 = κ.alpha*z^κ.gamma
    v2 = -κ.beta
    v2 * v1 * ((v1 + 1)^(v2 - 1)) * log(z)
end

function kappa_dp{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, param::Symbol, z::T)
    if param == :alpha
        kappa_dalpha(κ, z)
    elseif param == :beta
        kappa_dbeta(κ, z)
    elseif param == :gamma
        kappa_dgamma(κ, z)
    else
        zero(T)
    end
end


#==========================================================================
  Power Kernel
==========================================================================#

immutable PowerKernel{T<:FloatingPoint,CASE} <: SquaredDistanceKernel{T}
    gamma::T
    function PowerKernel(γ::T)
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))
        if CASE == :γ1 && γ != 1
            warn("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(γ)
    end
end
PowerKernel{T<:FloatingPoint}(γ::T = 1.0) = PowerKernel{T, γ == 1 ? :γ1 : :Ø}(γ)

ismercer(::PowerKernel) = false

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
    Boughorbel, S.; Tarel, J.-P.; Nozha Boujemaa, Conditionally 
    Positive Definite Kernels for SVM Based Image Recognition, 
    Multimedia and Expo, 2005. ICME 2005. IEEE International Conference 
    on , vol., no., pp.113,116, 6-6 July 2005
    """
end

kappa{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = -z^(κ.gamma)
kappa{T<:FloatingPoint}(κ::PowerKernel{T,:γ1}, z::T) = -z

kappa_dz{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = -κ.gamma*(z^(κ.gamma - 1))
kappa_dz{T<:FloatingPoint}(κ::PowerKernel{T,:γ1}, z::T) = -one(T)

kappa_dz2{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = (κ.gamma - κ.gamma^2)*(z^(κ.gamma - 2))
kappa_dz2{T<:FloatingPoint}(κ::PowerKernel{T,:γ1}, z::T) = zero(T)

kappa_dgamma{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = -log(z)*(z^(κ.gamma))

kappa_dp{T<:FloatingPoint}(κ::PowerKernel{T}, param::Symbol, z::T) = param == :gamma ? kappa_dgamma(κ, z) : zero(T)


#==========================================================================
  Log Kernel
==========================================================================#

immutable LogKernel{T<:FloatingPoint,CASE} <: SquaredDistanceKernel{T}
    alpha::T
    gamma::T
    function LogKernel(α::T, γ::T)
        α > 0 || throw(ArgumentError("α = $(α) must be greater than zero."))
        0 < γ <= 1 || throw(ArgumentError("γ = $(γ) must be in the interval (0,1]."))
        if CASE == :γ1 && γ != 1
            warn("Special case γ = 1 flagged but γ = $(γ)")
        end
        new(α,γ)
    end
end
LogKernel{T<:FloatingPoint}(α::T = 1.0, γ::T = one(T)) = LogKernel{T, γ == 1 ? :γ1 : :Ø}(α, γ)

ismercer(::LogKernel) = false

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

kappa{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = -log(κ.alpha*z^(κ.gamma) + 1)
kappa{T<:FloatingPoint}(κ::LogKernel{T,:γ1}, z::T) = -log(κ.alpha*z + 1)

kappa_dz{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = - κ.alpha * κ.gamma * z^(κ.gamma-1) / (κ.alpha*z^(κ.gamma) + 1)
kappa_dz{T<:FloatingPoint}(κ::LogKernel{T,:γ1}, z::T) = - κ.alpha / (κ.alpha*z + 1)

function kappa_dz2{T<:FloatingPoint}(κ::LogKernel{T}, z::T)
    v1 = κ.alpha*z^(κ.gamma)
    -κ.alpha * κ.gamma * (z^(κ.gamma-2)) * (κ.gamma - 1 - v1) / (v1 + 1)^2
end

kappa_dalpha{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = - z^(κ.gamma) / (κ.alpha*z^(κ.gamma) + 1)

kappa_dgamma{T<:FloatingPoint}(κ::LogKernel{T}, z::T) = - log(z) * κ.alpha * z^(κ.gamma) / (κ.alpha*z^(κ.gamma) + 1)

function kappa_dp{T<:FloatingPoint}(κ::LogKernel{T}, param::Symbol, z::T)
    if param == :alpha
        kappa_dalpha(κ, z)
    elseif param == :gamma
        kappa_dgamma(κ, z)
    else
        zero(T)
    end
end
