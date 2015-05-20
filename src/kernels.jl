#===================================================================================================
  Generic Kernels
===================================================================================================#

abstract Kernel{T<:FloatingPoint}

eltype{T}(κ::Kernel{T}) = T

#call{T<:FloatingPoint}(κ::Kernel{T}, x::Vector{T}, y::Vector{T}) = kernel_function(κ, x, y)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}) = kernel_matrix(κ, X)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}) = kernel_matrix(κ, X, Y)

isposdef(::Kernel) = false
iscondposdef(::Kernel) = false

abstract SimpleKernel{T<:FloatingPoint} <: Kernel{T}
abstract CompositeKernel{T<:FloatingPoint} <: Kernel{T}


#===================================================================================================
  Standard Kernels
===================================================================================================#

abstract StandardKernel{T<:FloatingPoint} <: SimpleKernel{T}

function show(io::IO, κ::StandardKernel)
    print(io, description_string(κ))
end

function description(io::IO, κ::StandardKernel)
    print(io, description_string_long(κ))
end
description(κ::StandardKernel) = description(STDOUT, κ)

kernelparameters(κ::StandardKernel) = names(κ) # default: all parameters of a kernel are scalars
# need to provide a more specific method if this doesn't apply

# concrete kernel types should provide kernel_dp(::<KernelType>, param::Symbol, x, y)
kernel_dp{T<:FloatingPoint}(κ::StandardKernel{T}, param::Integer, x::Array{T}, y::Array{T}) = kernel_dp(κ, kernelparameters(κ)[param], x, y)
kernel_dp{T<:FloatingPoint}(κ::StandardKernel{T}, param::Integer, x::T, y::T) = kernel_dp(κ, kernelparameters(κ)[param], x, y)


#===========================================================================
  Scalar Product Kernels - kernels of the form k(x,y) = κ(xᵀy)
===========================================================================#

abstract ScalarProductKernel{T<:FloatingPoint} <: StandardKernel{T}

kernel{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::Array{T}, y::Array{T}) = kappa(κ, scprod(x, y))
kernel{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::T, y::T) = kappa(κ, x*y)

function kernel_dx{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::Array{T}, y::Array{T}) # = kappa_dz(κ, scprod(x, y)) * scprod_dx(x, y)
    ∂κ_∂z = kappa_dz(κ, scprod(x, y))
    d = length(x)
    ∂k_∂x = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂x[i] = ∂κ_∂z * y[i]
    end
    ∂k_∂x
end
kernel_dy{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::Array{T}, y::Array{T}) = kernel_dx(κ, y, x)

function kernel_dxdy{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::Array{T}, y::Array{T})
    xᵀy = scprod(x, y)
    ∂κ_∂z = kappa_dz(κ, xᵀy)
    ∂κ²_∂z² = kappa_dz2(κ, xᵀy)
    d = length(x)
    ∂k²_∂x∂y = Array(T, d, d)
    @inbounds for j = 1:d
        for i = 1:d
            ∂k²_∂x∂y[i,j] = ∂κ²_∂z² * y[i] * x[j]
        end
        ∂k²_∂x∂y[j,j] += ∂κ_∂z
    end
    ∂k²_∂x∂y
end

function kernel_dxdy{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::T, y::T)
    xy = x * y
    kappa_dz2(κ, xy) * xy + kappa_dz(κ, xy)
end

kernel_dp{T<:FloatingPoint}(κ::ScalarProductKernel{T}, param::Symbol, x::Array{T}, y::Array{T}) = kappa_dp(κ, param, scprod(x, y))

# Scalar Product Kernel definitions
include("standardkernels/scalarproduct.jl")


#===========================================================================
  Squared Distance Kernels - kernels of the form k(x,y) = κ((x-y)ᵀ(x-y))
===========================================================================#

abstract SquaredDistanceKernel{T<:FloatingPoint} <: StandardKernel{T}

kernel{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::Array{T}, y::Array{T}) = kappa(κ, sqdist(x, y))
kernel{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::T, y::T) = kappa(κ, (x - y)^2)

function kernel_dx{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::Array{T}, y::Array{T}) # = kappa_dz(κ, sqdist(x, y)) * sqdist_dx(x, y)
    ∂κ_∂z = kappa_dz(κ, sqdist(x, y))
    d = length(x)
    ∂k_∂x = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂x[i] = 2∂κ_∂z * (x[i] - y[i])
    end
    ∂k_∂x
end
kernel_dy{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::Array{T}, y::Array{T}) = kernel_dx(κ, y, x)

kernel_dp{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, param::Symbol, x::Array{T}, y::Array{T}) = kappa_dp(κ, param, sqdist(x, y))

function kernel_dxdy{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::Array{T}, y::Array{T})
    ϵᵀϵ = sqdist(x, y)
    ∂κ_∂z = kappa_dz(κ, ϵᵀϵ)
    ∂κ²_∂z² = kappa_dz2(κ, ϵᵀϵ)
    d = length(x)
    ∂k²_∂x∂y = Array(T, d, d)
    @inbounds for j = 1:d
        ϵj = x[j] - y[j]
        for i = 1:d
            ϵi = x[i] - y[i]
            ∂k²_∂x∂y[i,j] = -4∂κ²_∂z² * ϵj * ϵi
        end
        ∂k²_∂x∂y[j,j] -= 2∂κ_∂z
    end
    ∂k²_∂x∂y
end

function kernel_dxdy{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::T, y::T)
    ϵᵀϵ = (x-y)^2
    -kappa_dz2(κ, ϵᵀϵ) * 4ϵᵀϵ - 2kappa_dz(κ, ϵᵀϵ)
end

# Squared Distance Kernel definitions
include("standardkernels/squareddistance.jl")


#===========================================================================
  Separable Kernels - kernels of the form k(x,y) = κ(x)ᵀκ(y)
===========================================================================#

abstract SeparableKernel{T<:FloatingPoint} <: StandardKernel{T}

function kappa_array!{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T})
    @inbounds for i = 1:length(x)
        x[i] = kappa(κ, x[i])
    end
    x
end

function kernel{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T}, y::Array{T})
    v = kappa_array!(κ, copy(x))
    z = kappa_array!(κ, copy(y))
    BLAS.dot(length(v), v, 1, z, 1)
end

function kappa_dz_array!{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T})
    @inbounds for i = 1:length(x)
        x[i] = kappa_dz(κ, x[i])
    end
    x
end

function kernel_dx{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T}, y::Array{T})
    v = kappa_dz_array!(κ, copy(x))
    z = kappa_array!(κ, copy(y))
    v.*z
end

function kernel_dy{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T}, y::Array{T})
    v = kappa_array!(κ, copy(x))
    z = kappa_dz_array!(κ, copy(y))
    v.*z
end

function kernel_dxdy{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T}, y::Array{T})
    v = kappa_dz_array!(κ, copy(x))
    z = kappa_dz_array!(κ, copy(y))
    diagm(v.*z)
end

function kernel_dp{T<:FloatingPoint}(κ::SeparableKernel{T}, param::Symbol, x::Array{T}, y::Array{T})
    (n = length(x)) == length(y) || error("Dimensions do not match")
    v = zero(T)
    @inbounds for i = 1:n
        v += kappa_dp(κ, param, x[i])*kappa(κ, y[i]) + kappa(κ, x[i])*kappa_dp(κ, param, y[i])
    end
    v
end


kernel{T<:FloatingPoint}(κ::SeparableKernel{T}, x::T, y::T) = kappa(κ, x) * kappa(κ, y) 

# Separable Kernel definitions
include("standardkernels/separable.jl")


#===========================================================================
  Automatic Relevance Determination (ARD) kernels
===========================================================================#

#include("standardkernels/ard.jl")

typealias ARDKernelTypes{T<:FloatingPoint} Union(SquaredDistanceKernel{T}, ScalarProductKernel{T})

immutable ARD{T<:FloatingPoint,K<:StandardKernel{T}} <: SimpleKernel{T} # let's not have an ARD{,ARD{,...{,Kernel}}}...
    k::K
    weights::Vector{T}
    function ARD(k::K, weights::Vector{T})
        isa(k, ARDKernelTypes) || throw(ArgumentError("ARD only implemented for $(join(ARDKernelTypes.body.types, ", ", " and "))"))
        all(weights .>= 0) || throw(ArgumentError("weights = $(weights) must all be >= 0."))
        new(k, weights)
    end
end

ARD{T<:FloatingPoint}(kernel::ARDKernelTypes{T}, weights::Vector{T}) = ARD{T,typeof(kernel)}(kernel, weights)
ARD{T<:FloatingPoint}(kernel::ARDKernelTypes{T}, dim::Integer) = ARD{T,typeof(kernel)}(kernel, ones(T, dim))

function description_string{T<:FloatingPoint,K<:StandardKernel}(κ::ARD{T,K}, eltype::Bool = true)
    "ARD" * (eltype ? "{$(T)}" : "") * "(kernel=$(description_string(κ.k, false)), weights=$(κ.weights))"
end

#=== ARD Squared Distance ===#

kernel{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kappa(κ.k, sqdist(x, y, κ.weights))

function kernel_dx{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    ∂κ_∂z = kappa_dz(κ.k, sqdist(x, y, w))
    d = length(x)
    ∂k_∂x = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂x[i] = 2∂κ_∂z * (x[i] - y[i]) * w[i]^2
    end
    ∂k_∂x
end
kernel_dy{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kernel_dx(κ, y, x)

function kernel_dw{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    ∂κ_∂z = kappa_dz(κ.k, sqdist(x, y, w))
    d = length(x)
    ∂k_∂w = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂w[i] = 2∂κ_∂z * (x[i] - y[i])^2 * w[i]
    end
    ∂k_∂w
end

function kernel_dxdy{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    ϵᵀW²ϵ = sqdist(x, y, w)
    ∂κ_∂z = kappa_dz(κ.k, ϵᵀW²ϵ)
    ∂κ²_∂z² = kappa_dz2(κ.k, ϵᵀW²ϵ)
    d = length(x)
    ∂k²_∂x∂y = Array(T, d, d)
    @inbounds for j = 1:d
        wj² = w[j]^2
        ϵj = (x[j] - y[j]) * wj²
        for i = 1:d
            ϵi = (x[i] - y[i]) * w[i]^2
            ∂k²_∂x∂y[i,j] = -4∂κ²_∂z² * ϵj * ϵi
        end
        ∂k²_∂x∂y[j,j] -= 2∂κ_∂z * wj²
    end
    ∂k²_∂x∂y
end

function kernel_dp{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, param::Symbol, x::Array{T}, y::Array{T})
    if param == :weights
        return kernel_dw(κ, x, y)
    else
        return kappa_dp(κ.k, param, sqdist(x, y, κ.weights))
    end
end

#=== ARD Scalar Product ===#

kernel{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kappa(κ.k, scprod(x, y, κ.weights))

function kernel_dx{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    ∂κ_∂z = kappa_dz(κ.k, scprod(x, y, w))
    d = length(x)
    ∂k_∂x = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂x[i] = ∂κ_∂z * y[i] * w[i]^2
    end
    ∂k_∂x
end
kernel_dy{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kernel_dx(κ, y, x)

function kernel_dw{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    ∂κ_∂z = kappa_dz(κ.k, scprod(x, y, w))
    d = length(x)
    ∂k_∂w = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂w[i] = 2∂κ_∂z * x[i] * y[i] * w[i]
    end
    ∂k_∂w
end

#= WIP
function kernel_dxdy{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    xᵀW²y = scprod(x, y, w)
    ∂κ_∂z = kappa_dz(κ.k, xᵀW²y)
    ∂κ²_∂z² = kappa_dz2(κ.k, xᵀW²y)
    d = length(x)
    ∂k²_∂x∂y = Array(T, d, d)
    @inbounds for j = 1:d
        wj² = w[j]^2
        v = ∂κ²_∂z² * x[j] * wj²
        for i = 1:d
            ∂k²_∂x∂y[i,j] = v * y[i]
        end
        ∂k²_∂x∂y[j,j] += ∂κ_∂z * wj²
    end
    ∂k²_∂x∂y
end
=#


#===================================================================================================
  Composite Kernels
===================================================================================================#

#===========================================================================
  Scaled Kernel
===========================================================================#

immutable ScaledKernel{T<:FloatingPoint} <: SimpleKernel{T}
    a::T
    k::StandardKernel{T}
    function ScaledKernel(a::T, κ::StandardKernel{T})
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ)
    end
end
ScaledKernel{T<:FloatingPoint}(a::T, κ::StandardKernel{T}) = ScaledKernel{T}(a, κ)

function ScaledKernel{T}(a::Real, κ::StandardKernel{T})
    U = promote_type(typeof(a), T)
    ScaledKernel(convert(U, a), convert(Kernel{U}, κ))
end

for kernel_type in (:ScaledKernel, :SimpleKernel, :Kernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, ψ::ScaledKernel) 
            ScaledKernel(convert(T, ψ.a), convert(Kernel{T}, ψ.k))
        end
    end
end

kernel{T<:FloatingPoint}(ψ::ScaledKernel{T}, x::Vector{T}, y::Vector{T}) = ψ.a * kernel(ψ.k, x, y)
kernel_dx{T<:FloatingPoint}(ψ::ScaledKernel{T}, x::Vector{T}, y::Vector{T}) = ψ.a * kernel_dx(ψ.k, x, y)
kernel_dy{T<:FloatingPoint}(ψ::ScaledKernel{T}, x::Vector{T}, y::Vector{T}) = ψ.a * kernel_dy(ψ.k, x, y)
kernel_dxdy{T<:FloatingPoint}(ψ::ScaledKernel{T}, x::Vector{T}, y::Vector{T}) = ψ.a * kernel_dxdy(ψ.k, x, y)

kernelparameters(κ::ScaledKernel) = vcat([:a], [symbol("k.$(param)") for param in kernelparameters(κ.k)])

function kernel_dp{T<:FloatingPoint}(ψ::ScaledKernel{T}, param::Symbol, x::Vector{T}, y::Vector{T})
    if param == :a
        kernel(ψ.k, x, y)
    elseif (sparam = string(param); beginswith(sparam, "k."))
        subparam = symbol(sparam[3:end])
        ψ.a * kernel_dp(ψ.k, subparam, x, y)
    else
        warn("derivative with respect to unrecognized symbol")
        zero(T)
    end
end

function kernel_dp{T<:FloatingPoint}(ψ::ScaledKernel{T}, param::Integer, x::Vector{T}, y::Vector{T})
    N = length(kernelparameters(ψ.k))
    if param == 1
        kernel_dp(ψ, :a, x, y)
    elseif 2 <= param <= N + 1
        ψ.a * kernel_dp(ψ.k, param-1, x, y)
    else
        throw(ArgumentError("param must be between 1 and $(N+1)"))
    end
end

function description_string{T<:FloatingPoint}(ψ::ScaledKernel{T})
    "ScaledKernel{$(T)}($(ψ.a)," * description_string(ψ.k, false) * ")"
end
isposdef(ψ::ScaledKernel) = isposdef(ψ.k)

function show(io::IO, ψ::ScaledKernel)
    print(io, description_string(ψ))
end

*(a::Real, κ::StandardKernel) = ScaledKernel(a, deepcopy(κ))
*(κ::StandardKernel, a::Real) = *(a, κ)

*(a::Real, ψ::ScaledKernel) = ScaledKernel(a * ψ.a, deepcopy(ψ.k))
*(ψ::ScaledKernel, a::Real) = *(a, ψ)


#===========================================================================
  Product Kernel
===========================================================================#

immutable KernelProduct{T<:FloatingPoint} <: CompositeKernel{T}
    a::T
    k1::Kernel{T}
    k2::Kernel{T}
    function KernelProduct(a::T, κ₁::Kernel{T}, κ₂::Kernel{T})
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ₁, κ₂)
    end
end
function KernelProduct{T<:FloatingPoint}(a::T, κ₁::Kernel{T}, κ₂::Kernel{T})
    KernelProduct{T}(a, κ₁, κ₂)
end

function KernelProduct{T,S}(a::Real, κ₁::Kernel{T}, κ₂::Kernel{S})
    U = promote_type(typeof(a), T, S)
    KernelProduct(convert(U, a), convert(Kernel{U}, κ₁), convert(Kernel{U}, κ₂))
end

for kernel_type in (:KernelProduct, :CompositeKernel, :Kernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, ψ::KernelProduct) 
            KernelProduct(convert(T, ψ.a), convert(Kernel{T}, ψ.k1), convert(Kernel{T}, ψ.k2))
        end
    end
end

function kernel{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T})
    ψ.a * kernel(ψ.k1, x, y) * kernel(ψ.k2, x, y)
end

function kernel_dx{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T})
    ψ.a * (kernel_dx(ψ.k1, x, y)*kernel(ψ.k2, x, y) + kernel(ψ.k1, x, y)*kernel_dx(ψ.k2, x, y))
end

function kernel_dy{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T})
    ψ.a * (kernel_dy(ψ.k1, x, y)*kernel(ψ.k2, x, y) + kernel(ψ.k1, x, y)*kernel_dy(ψ.k2, x, y))
end

function kernel_dxdy{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T})
    ψ.a * (kernel_dxdy(ψ.k1, x, y)*kernel(ψ.k2, x, y)
            + kernel_dy(ψ.k1, x, y)*kernel_dx(ψ.k2, x, y)'
            + kernel_dx(ψ.k1, x, y)*kernel_dy(ψ.k2, x, y)'
            + kernel(ψ.k1, x, y)*kernel_dxdy(ψ.k2, x, y))
end

kernelparameters(κ::KernelProduct) = vcat([:a], [symbol("k1.$(param)") for param in kernelparameters(κ.k)], [symbol("k2.$(param)") for param in kernelparameters(κ.k)])

function kernel_dp{T<:FloatingPoint}(ψ::KernelProduct{T}, param::Symbol, x::Vector{T}, y::Vector{T})
    if param == :a
        kernel(ψ.k1, x, y) * kernel(ψ.k2, x, y)
    elseif (sparam = string(param); beginswith(sparam, "k1."))
        subparam = symbol(sparam[4:end])
        ψ.a * kernel_dp(ψ.k1, subparam, x, y) * kernel(ψ.k2, x, y)
    elseif beginswith(sparam, "k2.")
        subparam = symbol(sparam[4:end])
        ψ.a * kernel(ψ.k1, x, y) * kernel_dp(ψ.k2, subparam, x, y)
    else
        warn("derivative with respect to unrecognized symbol")
        zero(T)
    end
end

function kernel_dp{T<:FloatingPoint}(ψ::KernelProduct{T}, param::Integer, x::Vector{T}, y::Vector{T})
    N1 = length(kernelparameters(ψ.k1))
    N2 = length(kernelparameters(ψ.k2))
    if param == 1
        kernel_dp(ψ, :a, x, y)
    elseif 2 <= param <= N1 + 1
        ψ.a * kernel_dp(ψ.k1, param-1, x, y) * kernel(ψ.k2, x, y)
    elseif N1 + 2 <= param <= N1 + N2 + 1
        ψ.a * kernel(ψ.k1, x, y) * kernel_dp(ψ.k2, param-N1-1, x, y)
    else
        throw(ArgumentError("param must be between 1 and $(N1+N2+1)"))
    end
end


isposdef(ψ::KernelProduct) = isposdef(ψ.k1) & isposdef(ψ.k2)

function description_string{T<:FloatingPoint}(ψ::KernelProduct{T}) 
    "KernelProduct{$(T)}($(ψ.a)," * description_string(ψ.k1, false) * "," * (
    description_string(ψ.k2, false) * ")")
end

function show(io::IO, ψ::KernelProduct)
    print(io, description_string(ψ))
end

function *{T,S}(κ₁::StandardKernel{T}, κ₂::StandardKernel{S})
    KernelProduct(one(promote_type(T, S)), deepcopy(κ₁), deepcopy(κ₂))
end

*(κ::StandardKernel, ψ::ScaledKernel) = KernelProduct(ψ.a, deepcopy(κ), deepcopy(ψ.k))
*(ψ::ScaledKernel, κ::StandardKernel) = KernelProduct(ψ.a, deepcopy(ψ.k), deepcopy(κ))

*(ψ₁::ScaledKernel, ψ₂::ScaledKernel) = KernelProduct(ψ₁.a*ψ₂.a, deepcopy(ψ₁.k), deepcopy(ψ₂.k))

*(a::Real, ψ::KernelProduct) = KernelProduct(a * ψ.a, deepcopy(ψ.k1), deepcopy(ψ.k2))
*(ψ::KernelProduct, a::Real) = *(a, ψ)


#===========================================================================
  Kernel Sum
===========================================================================#

immutable KernelSum{T<:FloatingPoint} <: CompositeKernel{T}
    a1::T
    k1::Kernel{T}
    a2::T
    k2::Kernel{T}
    function KernelSum(a₁::T, κ₁::Kernel{T}, a₂::T, κ₂::Kernel{T})
        a₁ > 0 || error("a₁ = $(a₁) must be greater than zero.")
        a₂ > 0 || error("a₂ = $(a₂) must be greater than zero.")
        new(a₁, κ₁, a₂, κ₂)
    end
end
function KernelSum{T<:FloatingPoint}(a₁::T, κ₁::Kernel{T}, a₂::T, κ₂::Kernel{T})
    KernelSum{T}(a₁, κ₁, a₂, κ₂)
end

function KernelSum{T,S}(a₁::Real, κ₁::Kernel{T}, a₂::Real, κ₂::Kernel{S})
    U = promote_type(typeof(a₁), typeof(a₂), T, S)
    KernelSum{U}(convert(U, a₁), convert(Kernel{U}, κ₁), convert(U, a₂), convert(Kernel{U}, κ₂))
end

for kernel_type in (:KernelSum, :CompositeKernel, :Kernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, ψ::KernelSum) 
            KernelSum(convert(T, ψ.a1), convert(Kernel{T}, ψ.k1), convert(T, ψ.a2), 
                          convert(Kernel{T}, ψ.k2))
        end
    end
end

function kernel{T<:FloatingPoint}(ψ::KernelSum{T}, x::Vector{T}, y::Vector{T})
    ψ.a1*kernel(ψ.k1, x, y) + ψ.a2*kernel(ψ.k2, x, y)
end

function kernel_dx{T<:FloatingPoint}(ψ::KernelSum{T}, x::Vector{T}, y::Vector{T})
    ψ.a1*kernel_dx(ψ.k1, x, y) + ψ.a2*kernel_dx(ψ.k2, x, y)
end

function kernel_dy{T<:FloatingPoint}(ψ::KernelSum{T}, x::Vector{T}, y::Vector{T})
    ψ.a1*kernel_dy(ψ.k1, x, y) + ψ.a2*kernel_dy(ψ.k2, x, y)
end

function kernel_dxdy{T<:FloatingPoint}(ψ::KernelSum{T}, x::Vector{T}, y::Vector{T})
    ψ.a1*kernel_dxdy(ψ.k1, x, y) + ψ.a2*kernel_dxdy(ψ.k2, x, y)
end

kernelparameters(κ::KernelSum) = vcat([:a1], [symbol("k1.$(param)") for param in kernelparameters(κ.k)], [:a2], [symbol("k2.$(param)") for param in kernelparameters(κ.k)])

function kernel_dp{T<:FloatingPoint}(ψ::KernelSum{T}, param::Symbol, x::Vector{T}, y::Vector{T})
    if param == :a1
        kernel(ψ.k1, x, y)
    elseif param == :a2
        kernel(ψ.k2, x, y)
    elseif (sparam = string(param); beginswith(sparam, "k1."))
        subparam = symbol(sparam[4:end])
        ψ.a1 * kernel_dp(ψ.k1, subparam, x, y)
    elseif beginswith(sparam, "k2.")
        subparam = symbol(sparam[4:end])
        ψ.a2 * kernel_dp(ψ.k2, subparam, x, y)
    else
        warn("derivative with respect to unrecognized symbol")
        zero(T)
    end
end

function kernel_dp{T<:FloatingPoint}(ψ::KernelSum{T}, param::Integer, x::Vector{T}, y::Vector{T})
    N1 = length(kernelparameters(ψ.k1))
    N2 = length(kernelparameters(ψ.k2))
    if param == 1
        kernel_dp(ψ, :a1, x, y)
    elseif 2 <= param <= N1 + 1
        ψ.a1 * kernel_dp(ψ.k1, param-1, x, y)
    elseif param == N1 + 2
        kernel_dp(ψ, :a2, x, y)
    elseif N1 + 3 <= param <= N1 + N2 + 2
        ψ.a2 * kernel_dp(ψ.k2, param-N1-2, x, y)
    else
        throw(ArgumentError("param must be between 1 and $(N1+N2+2)"))
    end
end

isposdef(ψ::KernelSum) = isposdef(ψ.k1) & isposdef(ψ.k2)

function description_string{T<:FloatingPoint}(ψ::KernelSum{T}) 
    "KernelSum{$(T)}($(ψ.a1)," * description_string(ψ.k1, false) * "," * "$(ψ.a2)," * description_string(ψ.k1, false) * ")"
end

function show(io::IO, ψ::KernelSum)
    print(io, description_string(ψ))
end

+(κ₁::StandardKernel, κ₂::StandardKernel) = KernelSum(1, deepcopy(κ₁), 1, deepcopy(κ₂))

+(κ::StandardKernel, ψ::ScaledKernel) = KernelSum(1, deepcopy(κ), ψ.a, deepcopy(ψ.k))
+(ψ::ScaledKernel, κ::StandardKernel) = KernelSum(ψ.a, deepcopy(ψ.k), 1, deepcopy(κ))

+(ψ₁::ScaledKernel, ψ₂::ScaledKernel) = KernelSum(ψ₁.a, deepcopy(ψ₁.k), ψ₂.a, deepcopy(ψ₂.k))

*(a::Real, ψ::KernelSum) = KernelSum(a*ψ.a1, deepcopy(ψ.k1), a*ψ.a2, deepcopy(ψ.k2))
*(ψ::KernelSum, a::Real) = *(a, ψ)
