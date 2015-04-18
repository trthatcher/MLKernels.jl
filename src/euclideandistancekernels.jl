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

# k(x,y) = f((x-y)ᵀ(x-y))
function kernel_function{T<:FloatingPoint}(κ::EuclideanDistanceKernel{T}, x::Vector{T},
                                           y::Vector{T})
    kernelize_scalar(κ, euclidean_distance(x, y))
end


#== Gaussian Kernel ===============#

immutable GaussianKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    η::T
    function GaussianKernel(η::T)
        η > 0 || throw(ArgumentError("σ = $(η) must be greater than 0."))
        new(η)
    end
end
GaussianKernel{T<:FloatingPoint}(η::T = 1.0) = GaussianKernel{T}(η)

function convert{T<:FloatingPoint}(::Type{GaussianKernel{T}}, κ::GaussianKernel) 
    GaussianKernel(convert(T, κ.η))
end

kernelize_scalar{T<:FloatingPoint}(κ::GaussianKernel{T}, ϵᵀϵ::T) = exp(-κ.η*ϵᵀϵ)

isposdef_kernel(::GaussianKernel) = true

function description_string{T<:FloatingPoint}(κ::GaussianKernel{T}, eltype::Bool = true) 
    "GaussianKernel" * (eltype ? "{$(T)}" : "") * "(η=$(κ.η))"
end

function description(κ::GaussianKernel)
    print(
        """ 
         Gaussian Kernel:
         
         The Gaussian kernel is a radial basis function based on the
         Gaussian distribution's probability density function. The feature
         has an infinite number of dimensions.

             k(x,y) = exp(-η‖x-y‖²)    x ∈ ℝⁿ, y ∈ ℝⁿ, η > 0

         Since the value of the function decreases as x and y differ, it can
         be interpretted as a similarity measure.
        """
    )
end


#== Laplacian Kernel ===============#

immutable LaplacianKernel{T<:FloatingPoint} <: EuclideanDistanceKernel{T}
    η::T
    function LaplacianKernel(η::T)
        η > 0 || throw(ArgumentError("η = $(η) must be greater than zero."))
        new(η)
    end
end
LaplacianKernel{T<:FloatingPoint}(η::T = 1.0) = LaplacianKernel{T}(η)

function convert{T<:FloatingPoint}(::Type{LaplacianKernel{T}}, κ::LaplacianKernel) 
    LaplacianKernel(convert(T, κ.η))
end

function kernelize_scalar{T<:FloatingPoint}(κ::LaplacianKernel{T}, ϵᵀϵ::T)
    exp(-κ.η*sqrt(ϵᵀϵ))
end

isposdef_kernel(κ::LaplacianKernel) = true

function description_string{T<:FloatingPoint}(κ::LaplacianKernel{T}, eltype::Bool = true) 
    "LaplacianKernel" * (eltype ? "{$(T)}" : "") * "(η=$(κ.η))"
end

function description(κ::LaplacianKernel)
    print(
        """ 
         Laplacian Kernel:
         
         The Laplacian (exponential) kernel is a radial basis function that
         differs from the Gaussian kernel in that it is a less sensitive
         similarity measure. Similarly, it is less sensitive to changes in
         the parameter η:

             k(x,y) = exp(-η‖x-y‖)    x ∈ ℝⁿ, y ∈ ℝⁿ, η > 0
        """
    )
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

isposdef_kernel(κ::RationalQuadraticKernel) = true

function description_string{T<:FloatingPoint}(κ::RationalQuadraticKernel{T}, eltype::Bool = true)
    "RationalQuadraticKernel" * (eltype ? "{$(T)}" : "") * "(c=$(κ.c))"
end

function description(κ::RationalQuadraticKernel)
    print(
        """ 
         Rational Quadratic Kernel:
         
         The rational quadratic kernel is a stationary kernel that is
         similar in shape to the Gaussian kernel:

             k(x,y) = 1 - ‖x-y‖²/(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
        """
    )
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

function description(κ::MultiQuadraticKernel)
    print(
        """ 
         Multi-Quadratic Kernel:
         
         The multi-quadratic kernel is a positive semidefinite kernel:

             k(x,y) = √(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
        """
    )
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

function description(κ::InverseMultiQuadraticKernel)
    print(
        """ 
         Inverse Multi-Quadratic Kernel:
         
         The inverse multi-quadratic kernel is a radial basis function. The
         resulting feature has an infinite number of dimensions:

             k(x,y) = 1/√(‖x-y‖² + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, c ≥ 0
        """
    )
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

function description(κ::PowerKernel)
    print(
        """ 
         Power Kernel:
         
         The power kernel (also known as the unrectified triangular kernel)
         is a positive semidefinite kernel. An important feature of the
         power kernel is that it is scale invariant. The function is given
         by:

             k(x,y) = -‖x-y‖ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, d > 0
        """
    )
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

function description(κ::LogKernel)
    print(
        """ 
         Log Kernel:
         
         The power kernel is a positive semidefinite kernel. The function is
         given by:

             k(x,y) = -log(‖x-y‖ᵈ + 1)    x ∈ ℝⁿ, y ∈ ℝⁿ, d > 0
        """
    )
end


#==========================================================================
  Conversions
==========================================================================#

for kernel in (:GaussianKernel, :LaplacianKernel, :RationalQuadraticKernel, :MultiQuadraticKernel, 
               :InverseMultiQuadraticKernel, :PowerKernel, :LogKernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{EuclideanDistanceKernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
        function convert{T<:FloatingPoint}(::Type{StandardKernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
        function convert{T<:FloatingPoint}(::Type{SimpleKernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
        function convert{T<:FloatingPoint}(::Type{Kernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
    end
end
