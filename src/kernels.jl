#===================================================================================================
  Generic Kernels
===================================================================================================#

typealias KernelInput{T} Union(T,Vector{T})

abstract Kernel{T<:FloatingPoint}

eltype{T}(κ::Kernel{T}) = T

#call{T<:FloatingPoint}(κ::Kernel{T}, x::Vector{T}, y::Vector{T}) = kernel_function(κ, x, y)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}) = kernel_matrix(κ, X)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}) = kernel_matrix(κ, X, Y)

ismercer(::Kernel) = false

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
  Periodic Kernel
===========================================================================#

include("standardkernels/periodic.jl")

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

function kernelparameters(κ::ARD)
    inner = kernelparameters(κ.k)
    if :weights in inner
        error("The inner kernel of ARD must not contain a 'weights' field.")
    end
    insert!(inner, 1, :weights)
end

kernel{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kappa(κ.k, sqdist(x, y, κ.weights))
kernel{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kappa(κ.k, scprod(x, y, κ.weights))

function kernel{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::T, y::T)
    if length(κ.weights) == 1
        kappa(κ.k, sqdist(x, y, κ.weights[1]))
    else
        throw(ArgumentError("Dimensions do not conform."))
    end
end
function kernel{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::T, y::T)
    if length(κ.weights) == 1
        kappa(κ.k, scprod(x, y, κ.weights[1]))
    else
        throw(ArgumentError("Dimensions do not conform."))
    end
end


#===========================================================================
  Kernel Conversions
===========================================================================#

for kernelobject in concretesubtypes(StandardKernel)
    kernelobjectname = kernelobject.name.name # symbol for concrete kernel type

    fieldconversions = [:(convert(T, κ.$field)) for field in names(kernelobject)]
    constructorcall = Expr(:call, kernelobjectname, fieldconversions...)

    @eval begin
        convert{T<:FloatingPoint}(::Type{$kernelobjectname{T}}, κ::$kernelobjectname) = $constructorcall
    end

    for kerneltype in supertypes(kernelobject)
        kerneltypename = kerneltype.name.name # symbol for abstract supertype

        @eval begin
            function convert{T<:FloatingPoint}(::Type{$kerneltypename{T}}, κ::$kernelobjectname)
                convert($kernelobjectname{T}, κ)
            end
        end
    end
end


#===================================================================================================
  Composite Kernels
===================================================================================================#

#===========================================================================
  Product Kernel
===========================================================================#

immutable KernelProduct{T<:FloatingPoint} <: CompositeKernel{T}
    a::T
    k::Vector{Kernel{T}}
    function KernelProduct(a::T, κ::Vector{Kernel{T}})
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ)
    end
end
KernelProduct{T<:FloatingPoint}(a::T, κ::Vector{Kernel{T}}) = KernelProduct{T}(a, κ)

function KernelProduct(a::Real, κ::Kernel...)
    U = promote_type(typeof(a), map(eltype, κ)...)
    KernelProduct{U}(convert(U, a), Kernel{U}[κ...])
end

for kernel_type in (:KernelProduct, :CompositeKernel, :Kernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, ψ::KernelProduct)
            KernelProduct(convert(T, ψ.a), Kernel{T}[ψ.k...])
        end
    end
end

kernel{T<:FloatingPoint}(ψ::KernelProduct{T}, x::KernelInput{T}, y::KernelInput{T}) = ψ.a * prod(map(κ -> kernel(κ,x,y), ψ.k))

function kernelparameters(ψ::KernelProduct)
    parameter_list = [:a]
    for i = 1:length(ψ.k)
        append!(parameter_list, [symbol("k[$(i)].$(θ)") for θ in kernelparameters(ψ.k[i])])
    end
    parameter_list
end

ismercer(ψ::KernelProduct) = all(ismercer, ψ.k)

function description_string{T<:FloatingPoint}(ψ::KernelProduct{T}, eltype::Bool = true)
    descs = map(κ -> description_string(κ, false), ψ.k)
    if eltype
        "KernelProduct" * (eltype ? "{$(T)}" : "") * "($(ψ.a), $(join(descs, ", ")))"
    else
        (ψ.a == 1 ? "" : "$(ψ.a)") * "($(join(descs, " * ")))"
    end
end

function show(io::IO, ψ::KernelProduct)
    print(io, description_string(ψ))
end

*(a::Real, κ::Kernel) = KernelProduct(a, κ)
*(κ::Kernel, a::Real) = *(a, κ)

*(a::Real, ψ::KernelProduct) = KernelProduct(a * ψ.a, ψ.k...)
*(ψ::KernelProduct, a::Real) = *(a, ψ)

*(κ1::KernelProduct, κ2::KernelProduct) = KernelProduct(κ1.a * κ2.a, κ1.k..., κ2.k...)

*(κ::Kernel, ψ::KernelProduct) = KernelProduct(ψ.a, κ, ψ.k...)
*(ψ::KernelProduct, κ::Kernel) = KernelProduct(ψ.a, ψ.k..., κ)

*(κ1::Kernel, κ2::Kernel) = KernelProduct(1, κ1, κ2)


#===========================================================================
  Kernel Sum
===========================================================================#

immutable KernelSum{T<:FloatingPoint} <: CompositeKernel{T}
    k::Vector{Kernel{T}}
end

function KernelSum(κ::Kernel...)
    U = promote_type(map(eltype, κ)...)
    KernelSum{U}(Kernel{U}[κ...])
end

for kernel_type in (:KernelSum, :CompositeKernel, :Kernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, ψ::KernelSum)
            KernelSum(Kernel{T}[ψ.k...])
        end
    end
end

kernel{T<:FloatingPoint}(ψ::KernelSum{T}, x::KernelInput{T}, y::KernelInput{T}) = sum(map(κ -> kernel(κ,x,y), ψ.k))

function kernelparameters(ψ::KernelSum)
    parameter_list = Symbol[]
    for i = 1:length(ψ.k)
        append!(parameter_list, [symbol("k[$(i)].$(θ)") for θ in kernelparameters(ψ.k[i])])
    end
    parameter_list
end

ismercer(ψ::KernelSum) = all(ismercer, ψ.k)

function description_string{T<:FloatingPoint}(ψ::KernelSum{T}, eltype::Bool = true)
    descs = map(κ -> description_string(κ, false), ψ.k)
    if eltype
        "KernelSum" * (eltype ? "{$(T)}" : "") * "($(join(descs, ", ")))"
    else
        "($(join(descs, " + ")))"
    end
end

function show(io::IO, ψ::KernelSum)
    print(io, description_string(ψ))
end

+(ψ1::KernelSum, ψ2::KernelSum) = KernelSum(ψ1.k..., ψ2.k...)

+(κ::Kernel, ψ::KernelSum) = KernelSum(κ, ψ.k...)
+(ψ::KernelSum, κ::Kernel) = KernelSum(ψ.k..., κ)

+(κ1::Kernel, κ2::Kernel) = KernelSum(κ1, κ2)
