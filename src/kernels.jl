#===================================================================================================
  Generic Kernels
===================================================================================================#

abstract Kernel{T<:FloatingPoint}

eltype{T}(κ::Kernel{T}) = T

#call{T<:FloatingPoint}(κ::Kernel{T}, x::Vector{T}, y::Vector{T}) = kernel_function(κ, x, y)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}) = kernel_matrix(κ, X)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}) = kernel_matrix(κ, X, Y)

isposdef(::Kernel) = false
iscondposdef(κ::Kernel) = isposdef(κ)

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


#===========================================================================
  Scalar Product Kernels - kernels of the form k(x,y) = κ(xᵀy)
===========================================================================#

abstract ScalarProductKernel{T<:FloatingPoint} <: StandardKernel{T}

kernel{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::Array{T}, y::Array{T}) = kappa(κ, scprod(x, y))
kernel{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::T, y::T) = kappa(κ, x*y)

# Scalar Product Kernel definitions
include("standardkernels/scalarproduct.jl")


#===========================================================================
  Squared Distance Kernels - kernels of the form k(x,y) = κ((x-y)ᵀ(x-y))
===========================================================================#

abstract SquaredDistanceKernel{T<:FloatingPoint} <: StandardKernel{T}

kernel{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::Array{T}, y::Array{T}) = kappa(κ, sqdist(x, y))
kernel{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::T, y::T) = kappa(κ, (x - y)^2)

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
kernel{T<:FloatingPoint}(κ::SeparableKernel{T}, x::T, y::T) = kappa(κ, x) * kappa(κ, y) 

# Separable Kernel definitions
include("standardkernels/separable.jl")


#===========================================================================
  Automatic Relevance Determination (ARD) kernels
===========================================================================#

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

kernel{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kappa(κ.k, sqdist(x, y, κ.weights))
kernel{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kappa(κ.k, scprod(x, y, κ.weights))


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

kernelparameters(κ::ScaledKernel) = vcat([:a], [symbol("k.$(param)") for param in kernelparameters(κ.k)])

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

kernelparameters(κ::KernelProduct) = vcat([:a], [symbol("k1.$(param)") for param in kernelparameters(κ.k1)], [symbol("k2.$(param)") for param in kernelparameters(κ.k2)])

isposdef(ψ::KernelProduct) = isposdef(ψ.k1) & isposdef(ψ.k2)

function description_string{T<:FloatingPoint}(ψ::KernelProduct{T}) 
    "KernelProduct{$(T)}($(ψ.a)," * description_string(ψ.k1, false) * "," * description_string(ψ.k2, false) * ")"
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

kernelparameters(κ::KernelSum) = vcat([:a1], [symbol("k1.$(param)") for param in kernelparameters(κ.k1)], [:a2], [symbol("k2.$(param)") for param in kernelparameters(κ.k2)])

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
