#===================================================================================================
  Euclidean Distance Kernels
===================================================================================================#

abstract EuclideanDistanceKernel{T<:FloatingPoint} <: StandardKernel{T}

# ϵᵀϵ = (x-y)ᵀ(x-y)
function euclidean_distance{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    n = length(x)
    ϵ = BLAS.axpy!(n, -one(T), y, 1, copy(x), 1)
    BLAS.dot(n, ϵ, 1, ϵ, 1)
end

function deuclidean_distance_dx{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    2(x-y)
end

function deuclidean_distance_dy{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    2(y-x)
end

# k(x,y) = f((x-y)ᵀ(x-y))
function kernel{T<:FloatingPoint}(κ::EuclideanDistanceKernel{T}, x::Array{T}, y::Array{T})
    kernelize_scalar(κ, euclidean_distance(x, y))
end

function dkernel_dx{T<:FloatingPoint}(κ::EuclideanDistanceKernel{T}, x::Array{T}, y::Array{T})
    kernelize_scalar_deriv(κ, euclidean_distance(x, y)) * deuclidean_distance_dx(x, y)
end

function dkernel_dy{T<:FloatingPoint}(κ::EuclideanDistanceKernel{T}, x::Array{T}, y::Array{T})
    kernelize_scalar_deriv(κ, euclidean_distance(x, y)) * deuclidean_distance_dy(x, y)
end

function d2kernel_dxdy{T<:FloatingPoint}(κ::EuclideanDistanceKernel{T}, x::Array{T}, y::Array{T})
    ϵᵀϵ = euclidean_distance(x, y)
    -kernelize_scalar_deriv2(κ, ϵᵀϵ) * 4(x-y)*(x-y)' - 2kernelize_scalar_deriv(κ, ϵᵀϵ)*eye(length(x))
end

function dkernel_dp{T<:FloatingPoint}(κ::EuclideanDistanceKernel{T}, param::Union(Integer,Symbol), x::Array{T}, y::Array{T})
    kernelize_scalar_pderiv(κ, param, euclidean_distance(x, y))
end

function kernel{T<:FloatingPoint}(κ::EuclideanDistanceKernel{T}, x::T, y::T)
    kernelize_scalar(κ, (x - y)^convert(T,2))
end

#== Gaussian Kernel ===============#

immutable GaussianKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
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

function kernelize_scalar{T<:FloatingPoint}(κ::GaussianKernel{T}, ϵᵀϵ::T)
    exp(ϵᵀϵ/(convert(T,-2)*(κ.sigma^convert(T,2))))
end

function kernelize_scalar_deriv{T<:FloatingPoint}(κ::GaussianKernel{T}, ϵᵀϵ::T)
    kernelize_scalar(κ, ϵᵀϵ)/(convert(T,-2)*(κ.sigma^convert(T,2)))
end

function kernelize_scalar_deriv2{T<:FloatingPoint}(κ::GaussianKernel{T}, ϵᵀϵ::T)
    kernelize_scalar(κ, ϵᵀϵ)/(convert(T,4)*(κ.sigma^convert(T,4)))
end

function kernelize_scalar_pderiv{T<:FloatingPoint}(κ::GaussianKernel{T}, param::Symbol, ϵᵀϵ::T)
    if param == :sigma
        kernelize_scalar(κ, ϵᵀϵ) * ϵᵀϵ * κ.sigma^convert(T,-3)
    else
        zero(T)
    end
end

function kernelize_scalar_pderiv{T<:FloatingPoint}(κ::GaussianKernel{T}, param::Integer, ϵᵀϵ::T)
    if param == 1
        kernelize_scalar_pderiv(κ, :sigma, ϵᵀϵ)
    else
        throw(ArgumentError("param must be 1"))
    end
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

immutable LaplacianKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
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

function kernelize_scalar{T<:FloatingPoint}(κ::LaplacianKernel{T}, ϵᵀϵ::T)
    exp(sqrt(ϵᵀϵ)/(-κ.sigma))
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

immutable RationalQuadraticKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
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

function kernelize_scalar{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, ϵᵀϵ::T)
    one(T) - ϵᵀϵ/(ϵᵀϵ + κ.c)
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

immutable MultiQuadraticKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
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

function kernelize_scalar{T<:FloatingPoint}(κ::MultiQuadraticKernel{T}, ϵᵀϵ::T)
    sqrt(ϵᵀϵ + κ.c)
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

immutable InverseMultiQuadraticKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    c::T
    function InverseMultiQuadraticKernel(c::T)
        c > 0 || throw(ArgumentError("c = $(c) must be greater than zero."))
        new(c)
    end
end
InverseMultiQuadraticKernel{T<:FloatingPoint}(c::T = 1.0) = InverseMultiQuadraticKernel{T}(c)

function convert{T<:FloatingPoint}(::Type{InverseMultiQuadraticKernel{T}},
                                   κ::InverseMultiQuadraticKernel)
    InverseMultiQuadraticKernel(convert(T, κ.c))
end

function kernelize_scalar{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T}, ϵᵀϵ::T)
    one(T) / sqrt(ϵᵀϵ + κ.c)
end

function description_string{T<:FloatingPoint}(κ::InverseMultiQuadraticKernel{T},
                                              eltype::Bool = true)
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

immutable PowerKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    d::T
    function PowerKernel(d::T)
        d > 0 || throw(ArgumentError("d = $(d) must be a positive integer."))
        b = trunc(d)
        d == b || warn("d = $(d) was truncated to $(b).")
        new(b)
    end
end
PowerKernel{T<:FloatingPoint}(d::T = 2.0) = PowerKernel{T}(d)
PowerKernel(d::Integer) = PowerKernel(convert(Float64, d))

convert{T<:FloatingPoint}(::Type{PowerKernel{T}}, κ::PowerKernel) = PowerKernel(convert(T, κ.d))

kernelize_scalar{T<:FloatingPoint}(κ::PowerKernel{T}, ϵᵀϵ::T) = -sqrt(ϵᵀϵ)^(κ.d)

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

immutable LogKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    d::T
    function LogKernel(d::T)
        d > 0 || throw(ArgumentError("d = $(d) must be a positive integer."))
        b = trunc(d)
        d == b || warn("d = $(d) was truncated to $(b).")
        new(b)
    end
end
LogKernel{T<:FloatingPoint}(d::T = 1.0) = LogKernel{T}(d)
LogKernel(d::Integer) = LogKernel(convert(Float32, d))

convert{T<:FloatingPoint}(::Type{LogKernel{T}}, κ::LogKernel) = LogKernel(convert(T, κ.d))

function kernelize_scalar{T<:FloatingPoint}(κ::LogKernel{T}, ϵᵀϵ::T)
    -log(sqrt(ϵᵀϵ)^(κ.d) + one(T))
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


#==========================================================================
  Conversions
==========================================================================#

for kernelobject in (:GaussianKernel, :LaplacianKernel, :RationalQuadraticKernel,
               :MultiQuadraticKernel, :InverseMultiQuadraticKernel,
               :PowerKernel, :LogKernel)
    for kerneltype in (:EuclideanDistanceKernel, :StandardKernel, :SimpleKernel, :Kernel)
        @eval begin
            function convert{T<:FloatingPoint}(::Type{$kerneltype{T}}, κ::$kernelobject)
                convert($kernelobject{T}, κ)
            end
        end
    end
end
