#===================================================================================================
  Squared Distance Kernel Definitions: z = ϵᵀϵ
===================================================================================================#

#== Gaussian Kernel ===============#

immutable GaussianKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    sigma::T
    function GaussianKernel(σ::T)
        σ > 0 || throw(ArgumentError("σ = $(σ) must be greater than 0."))
        new(σ)
    end
end
GaussianKernel{T<:FloatingPoint}(σ::T = 1.0) = GaussianKernel{T}(σ)
SquaredExponentialKernel{T<:FloatingPoint}(l::T = 1.0) = GaussianKernel{T}(l)

function convert{T<:FloatingPoint}(::Type{GaussianKernel{T}}, κ::GaussianKernel)
    GaussianKernel(convert(T, κ.sigma))
end

kappa{T<:FloatingPoint}(κ::GaussianKernel{T}, z::T) = exp(z/(-2κ.sigma^2))
kappa_dz{T<:FloatingPoint}(κ::GaussianKernel{T}, z::T, kz = kappa(κ, z))  = kz / (-2κ.sigma^2)
kappa_dz2{T<:FloatingPoint}(κ::GaussianKernel{T}, z::T, kz = kappa(κ, z)) = kz / (4κ.sigma^4)
kappa_dsigma{T<:FloatingPoint}(κ::GaussianKernel{T}, z::T, kz=kappa(κ, z)) = kz * z * κ.sigma^(-3)

function kappa_dp{T<:FloatingPoint}(κ::GaussianKernel{T}, param::Symbol, z::T)
    param == :sigma ? kappa_dsigma(κ, z) : zero(T)
end

isposdef(::GaussianKernel) = true

function description_string{T<:FloatingPoint}(κ::GaussianKernel{T}, eltype::Bool = true)
    "GaussianKernel" * (eltype ? "{$(T)}" : "") * "(σ=$(κ.sigma))"
end

function description_string_long(::GaussianKernel)
    """
    Gaussian Kernel:
    
    The Gaussian kernel is a radial basis function based on the
    Gaussian distribution's probability density function. The feature
    has an infinite number of dimensions.
    
        k(x,y) = exp(-‖x-y‖²/(2σ²))    x ∈ ℝⁿ, y ∈ ℝⁿ, σ > 0
    
    Since the value of the function decreases as x and y differ, it can
    be interpreted as a similarity measure.
    """
end


#== Laplacian Kernel ===============#

immutable LaplacianKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    sigma::T
    function LaplacianKernel(σ::T)
        σ > 0 || throw(ArgumentError("σ = $(σ) must be greater than zero."))
        new(σ)
    end
end
LaplacianKernel{T<:FloatingPoint}(σ::T = 1.0) = LaplacianKernel{T}(σ)
ExponentialKernel{T<:FloatingPoint}(σ::T = 1.0) = LaplacianKernel{T}(σ)

function convert{T<:FloatingPoint}(::Type{LaplacianKernel{T}}, κ::LaplacianKernel)
    LaplacianKernel(convert(T, κ.sigma))
end

kappa{T<:FloatingPoint}(κ::LaplacianKernel{T}, z::T) = exp(sqrt(z)/(-κ.sigma))
kappa_dz{T<:FloatingPoint}(κ::LaplacianKernel{T}, z::T, kz = kappa(κ, z)) = kz/(-2κ.sigma*sqrt(z))
kappa_dz2{T<:FloatingPoint}(κ::LaplacianKernel{T}, z::T, kz = kappa(κ, z)) = kz*(κ.sigma + sqrt(z))/(4κ.sigma^2 * z^(3/2))
kappa_dsigma{T<:FloatingPoint}(κ::LaplacianKernel{T}, z::T, kz = kappa(κ, z)) = kz * sqrt(z) / κ.sigma^2

function kappa_dp{T<:FloatingPoint}(κ::LaplacianKernel{T}, param::Symbol, z::T)
    param == :sigma ? kappa_dsigma(κ, z) : zero(T)
end

isposdef(::LaplacianKernel) = true

function description_string{T<:FloatingPoint}(κ::LaplacianKernel{T}, eltype::Bool = true)
    "LaplacianKernel" * (eltype ? "{$(T)}" : "") * "(σ=$(κ.sigma))"
end

function description_string_long(::LaplacianKernel)
    """
    Laplacian Kernel:
    
    The Laplacian (exponential) kernel is a radial basis function that
    differs from the Gaussian kernel in that it is a less sensitive
    similarity measure. Similarly, it is less sensitive to changes in
    the parameter σ:

        k(x,y) = exp(-‖x-y‖/σ)    x ∈ ℝⁿ, y ∈ ℝⁿ, σ > 0
    """
end


#== Rational Quadratic Kernel ===============#

immutable RationalQuadraticKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    c::T
    function RationalQuadraticKernel(c::T)
        c > 0 || throw(ArgumentError("c = $(c) must be greater than zero."))
        new(c)
    end
end
RationalQuadraticKernel{T<:FloatingPoint}(c::T = 1.0) = RationalQuadraticKernel{T}(c)

function convert{T<:FloatingPoint}(::Type{RationalQuadraticKernel{T}}, κ::RationalQuadraticKernel)
    RationalQuadraticKernel(convert(T, κ.c))
end

kappa{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = 1 - z/(z + κ.c)
kappa_dz{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = -κ.c/((z + κ.c)^2)
kappa_dz2{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = 2κ.c/((z + κ.c)^3)
kappa_dc{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, z::T) = z/((z + κ.c)^2)

function kappa_dp{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, param::Symbol, z::T)
    param == :c ? kappa_dc(κ, z) : zero(T)
end

isposdef(::RationalQuadraticKernel) = true

function description_string{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, eltype::Bool = true)
    "RationalQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(c=$(κ.c))"
end

function description_string_long(::RationalQuadraticKernel)
    """
    Rational Quadratic Kernel:
    
    The rational quadratic kernel is a stationary kernel that is
    similar in shape to the Gaussian kernel:
    
        k(x,y) = 1 - ‖x-y‖²/(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
    """
end


#== Multi-Quadratic Kernel ===============#

immutable MultiQuadraticKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    c::T
    function MultiQuadraticKernel(c::T)
        c > 0 || throw(ArgumentError("c = $(c) must be greater than zero."))
        new(c)
    end
end
MultiQuadraticKernel{T<:FloatingPoint}(c::T = 1.0) = MultiQuadraticKernel{T}(c)

function convert{T<:FloatingPoint}(::Type{MultiQuadraticKernel{T}}, κ::MultiQuadraticKernel)
    MultiQuadraticKernel(convert(T, κ.c))
end

kappa{T<:FloatingPoint}(κ::MultiQuadraticKernel{T}, z::T) = sqrt(z + κ.c)
kappa_dz{T<:FloatingPoint}(κ::MultiQuadraticKernel{T}, z::T) = 1/(2sqrt(z + κ.c))
kappa_dz2{T<:FloatingPoint}(κ::MultiQuadraticKernel{T}, z::T) = -1/(4(z + κ.c)^(3/2))
kappa_dc{T<:FloatingPoint}(κ::MultiQuadraticKernel{T}, z::T) = 1/(2sqrt(z + κ.c))

function kappa_dp{T<:FloatingPoint}(κ::MultiQuadraticKernel{T}, param::Symbol, z::T)
    param == :c ? kappa_dc(κ, z) : zero(T)
end

function description_string{T<:FloatingPoint}(κ::MultiQuadraticKernel{T}, eltype::Bool = true)
    "MultiQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(c=$(κ.c))"
end

function description_string_long(::MultiQuadraticKernel)
    """
    Multi-Quadratic Kernel:
    
    The multi-quadratic kernel is a positive semidefinite kernel:
    
        k(x,y) = √(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
    """
end


#== Inverse Multi-Quadratic Kernel ===============#

immutable InverseMultiQuadraticKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    c::T
    function InverseMultiQuadraticKernel(c::T)
        c > 0 || throw(ArgumentError("c = $(c) must be greater than zero."))
        new(c)
    end
end
InverseMultiQuadraticKernel{T<:FloatingPoint}(c::T = 1.0) = InverseMultiQuadraticKernel{T}(c)

function convert{T<:FloatingPoint}(::Type{InverseMultiQuadraticKernel{T}}, κ::InverseMultiQuadraticKernel)
    InverseMultiQuadraticKernel(convert(T, κ.c))
end

kappa{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T}, z::T) = 1 / sqrt(z + κ.c)
kappa_dz{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T}, z::T) = -1/(2(z + κ.c)^(3/2))
kappa_dz2{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T}, z::T) = 3/(4(z + κ.c)^(5/2))
kappa_dc{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T}, z::T) = -1/(2(z + κ.c)^(3/2))

function kappa_dp{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T}, param::Symbol, z::T)
    param == :c ? kappa_dc(κ, z) : zero(T)
end

function description_string{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T}, eltype::Bool = true)
    "InverseMultiQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(c=$(κ.c))"
end

function description_string_long(::InverseMultiQuadraticKernel)
    """
    Inverse Multi-Quadratic Kernel:
    
    The inverse multi-quadratic kernel is a radial basis function. The
    resulting feature has an infinite number of dimensions:
    
        k(x,y) = 1/√(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
    """
end


#== Power Kernel ===============#

immutable PowerKernel{T<:FloatingPoint} <: SquaredDistanceKernel{T}
    d::T
    function PowerKernel(d::T)
        d > 0 || throw(ArgumentError("d = $(d) must be a greater than zero."))
        new(d)
    end
end
PowerKernel{T<:FloatingPoint}(d::T = 2.0) = PowerKernel{T}(d)
PowerKernel(d::Integer) = PowerKernel(convert(Float64, d))

convert{T<:FloatingPoint}(::Type{PowerKernel{T}}, κ::PowerKernel) = PowerKernel(convert(T, κ.d))

kappa{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = -z^(κ.d/2)
kappa_dz{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = (-κ.d/2)*(z^(κ.d/2 - 1))
kappa_dz2{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = -((κ.d^2 - 2κ.d)/4)*(z^(κ.d/2 - 2))
kappa_dd{T<:FloatingPoint}(κ::PowerKernel{T}, z::T) = -(log(z)/2)*(z^(κ.d/2))

function kappa_dp{T<:FloatingPoint}(κ::PowerKernel{T}, param::Symbol, z::T)
    param == :d ? kappa_dd(κ, z) : zero(T)
end

function description_string{T<:FloatingPoint}(κ::PowerKernel{T}, eltype::Bool = true)
    "PowerKernel" * (eltype ? "{$(T)}" : "") * "(d=$(κ.d))"
end

function description_string_long(::PowerKernel)
    """
    Power Kernel:
    
    The power kernel (also known as the unrectified triangular kernel)
    is a positive semidefinite kernel. An important feature of the
    power kernel is that it is scale invariant. The function is given
    by:
    
        k(x,y) = -‖x-y‖ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, d > 0
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

function description_string{T<:FloatingPoint}(κ::LogKernel{T}, eltype::Bool = true)
    "LogKernel" * (eltype ? "{$(T)}" : "") * "(d=$(κ.d))"
end

function description_string_long(::LogKernel)
    """
    Log Kernel:
    
    The log kernel is a positive semidefinite kernel. The function is
    given by:
    
        k(x,y) = -log(‖x-y‖ᵈ + 1)    x ∈ ℝⁿ, y ∈ ℝⁿ, d > 0
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
    "PeriodicKernel" * (eltype ? "{$(T)}" : "") * "(d=$(κ.d))"
end

function description_string_long(::PeriodicKernel)
    """
    Periodic Kernel:
    """
end


#==========================================================================
  Conversions
==========================================================================#

for kernelobject in (:GaussianKernel, :LaplacianKernel, :RationalQuadraticKernel,
               :MultiQuadraticKernel, :InverseMultiQuadraticKernel,
               :PowerKernel, :LogKernel)
    for kerneltype in (:SquaredDistanceKernel, :StandardKernel, :SimpleKernel, :Kernel)
        @eval begin
            function convert{T<:FloatingPoint}(::Type{$kerneltype{T}}, κ::$kernelobject)
                convert($kernelobject{T}, κ)
            end
        end
    end
end
