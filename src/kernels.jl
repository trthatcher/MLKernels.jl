#===================================================================================================
  Generic Kernels
===================================================================================================#

typealias KernelInput{T} Union(T,Array{T})

abstract Kernel{T<:FloatingPoint}

eltype{T}(κ::Kernel{T}) = T

#call{T<:FloatingPoint}(κ::Kernel{T}, x::Vector{T}, y::Vector{T}) = kernel_function(κ, x, y)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}) = kernel_matrix(κ, X)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}) = kernel_matrix(κ, X, Y)

ismercer(::Kernel) = false

abstract SimpleKernel{T<:FloatingPoint} <: Kernel{T}
abstract CompositeKernel{T<:FloatingPoint} <: Kernel{T}

KernelNode = Union(Expr, Symbol)

abstract KernelVariable{θ}

immutable BaseVariable{θ} <: KernelVariable{θ} end
BaseVariable(θ::Symbol) = BaseVariable{θ}()

function show{θ}(io::IO, variable::BaseVariable{θ})
    print(io, θ)
end

immutable SubVariable{θ} <: KernelVariable{θ}
    path::Vector{KernelNode}
end
SubVariable(path::Vector{KernelNode}, θ::Symbol) = SubVariable{θ}(path)

function generatepath{θ}(variable::SubVariable{θ})
    symbol(join([string(ex) for ex in variable.path], ".") * "." * string(θ))
end

function show(io::IO, θ::SubVariable)
    print(io, generatepath(θ))
end


#===================================================================================================
  Standard Kernels
===================================================================================================#

abstract StandardKernel{T<:FloatingPoint} <: SimpleKernel{T}

function show(io::IO, κ::Kernel)
    print(io, description_string(κ))
end

function description(io::IO, κ::StandardKernel)
    print(io, description_string_long(κ))
end
description(κ::StandardKernel) = description(STDOUT, κ)

kernelparameters(κ::StandardKernel) = [BaseVariable(θ) for θ in names(κ)]
kernelpath(path::Vector{KernelNode}, κ::StandardKernel) = [(path, θ) for θ in names(κ)]


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

function kernel{T<:FloatingPoint}(κ::SeparableKernel{T}, X::Array{T}, Y::Array{T})
    v = kappa_array!(κ, copy(x))
    z = kappa_array!(κ, copy(y))
    scprod(v, z)
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
    w::Vector{T}
    function ARD(κ::K, w::Vector{T})
        isa(κ, ARDKernelTypes) || throw(ArgumentError("ARD only implemented for $(join(ARDKernelTypes.body.types, ", ", " and "))"))
        all(w .>= 0) || throw(ArgumentError("weights = $(w) must all be >= 0."))
        new(κ, w)
    end
end

ARD{T<:FloatingPoint}(κ::ARDKernelTypes{T}, w::Vector{T}) = ARD{T,typeof(κ)}(κ, w)
ARD{T<:FloatingPoint}(κ::ARDKernelTypes{T}, dim::Integer) = ARD{T,typeof(κ)}(κ, ones(T, dim))

function description_string{T<:FloatingPoint,K<:StandardKernel}(κ::ARD{T,K}, eltype::Bool = true)
    "ARD" * (eltype ? "{$(T)}" : "") * "(κ=$(description_string(κ.k, false)), w=$(κ.w))"
end

kernelpath(path::Vector{KernelNode}, ψ::ARD) = append!([(path, :w)], kernelpath(KernelNode[path..., :k], ψ.k))

function kernelparameters(ψ::ARD)
    parameter_paths = append!([(KernelNode[], :w)], kernelpath(KernelNode[:k], ψ.k))
    KernelVariable[length(path) == 0 ? BaseVariable(θ) : SubVariable(path, θ) for (path, θ) in parameter_paths]
end

kernel{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kappa(κ.k, sqdist(x, y, κ.w))
kernel{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kappa(κ.k, scprod(x, y, κ.w))

function kernel{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::T, y::T)
    length(κ.w) == 1 || throw(ArgumentError("Dimensions do not conform."))
    kappa(κ.k, sqdist(x, y, κ.w[1]))
end

function kernel{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::T, y::T)
    length(κ.w) == 1 || throw(ArgumentError("Dimensions do not conform."))
    kappa(κ.k, scprod(x, y, κ.w[1]))
end


#===========================================================================
  Kernel Conversions
===========================================================================#

for kernelobject in concretesubtypes(StandardKernel)
    kernelobjectname = kernelobject.name.name  # symbol for concrete kernel type

    fieldconversions = [:(convert(T, κ.$field)) for field in names(kernelobject)]
    constructorcall = Expr(:call, kernelobjectname, fieldconversions...)

    @eval begin
        convert{T<:FloatingPoint}(::Type{$kernelobjectname{T}}, κ::$kernelobjectname) = $constructorcall
    end

    for kerneltype in supertypes(kernelobject)
        kerneltypename = kerneltype.name.name  # symbol for abstract supertype

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

function kernelpath(path::Vector{KernelNode}, ψ::CompositeKernel)
    parameter_list = [(path, :a)]
    for i = 1:length(ψ.k)
        append!(parameter_list, kernelpath(KernelNode[path..., :(k[$i])], ψ.k[i]))
    end
    parameter_list
end

function kernelparameters(ψ::CompositeKernel)
    parameter_paths = Any[(KernelNode[], :a)]
    for i = 1:length(ψ.k)
        append!(parameter_paths, kernelpath(KernelNode[:(k[$i])], ψ.k[i]))
    end
    KernelVariable[length(path) == 0 ? BaseVariable(θ) : SubVariable(path, θ) for (path, θ) in parameter_paths]
end


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

ismercer(ψ::KernelProduct) = all(ismercer, ψ.k)

function description_string{T<:FloatingPoint}(ψ::KernelProduct{T}, eltype::Bool = true)
    descs = map(κ -> description_string(κ, false), ψ.k)
    if eltype
        "KernelProduct" * (eltype ? "{$(T)}" : "") * "($(ψ.a), $(join(descs, ", ")))"
    else
        (ψ.a == 1 ? "" : "$(ψ.a)") * (length(descs)==1 ? descs[1] : "($(join(descs, " * ")))")
    end
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
    a::T
    k::Vector{Kernel{T}}
    function KernelSum(a::T, κ::Vector{Kernel{T}})
        a > 0 || error("a = $(a) must be greater than zero.")
        new(a, κ)
    end
end

function KernelSum(a::Real, κ::Kernel...)
    U = promote_type(typeof(a), map(eltype, κ)...)
    KernelSum{U}(convert(U, a), Kernel{U}[κ...])
end

for kernel_type in (:KernelSum, :CompositeKernel, :Kernel)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{$kernel_type{T}}, ψ::KernelSum)
            KernelSum(convert(T, ψ.a), Kernel{T}[ψ.k...])
        end
    end
end

kernel{T<:FloatingPoint}(ψ::KernelSum{T}, x::KernelInput{T}, y::KernelInput{T}) = sum(map(κ -> kernel(κ,x,y), ψ.k))

ismercer(ψ::KernelSum) = all(ismercer, ψ.k)

function description_string{T<:FloatingPoint}(ψ::KernelSum{T}, eltype::Bool = true)
    descs = map(κ -> description_string(κ, false), ψ.k)
    if eltype
        "KernelSum" * (eltype ? "{$(T)}" : "") * "($(join(descs, ", ")))"
    else
        "($(join(descs, " + ")))"
    end
end

+(a::Real, κ::Kernel) = KernelSum(a, κ)
+(κ::Kernel, a::Real) = +(a, κ)

+(a::Real, ψ::KernelSum) = KernelSum(a + ψ.a, ψ.k...)
+(ψ::KernelSum, a::Real) = +(a, ψ)

+(ψ1::KernelSum, ψ2::KernelSum) = KernelSum(ψ1.a + ψ2.a, ψ1.k..., ψ2.k...)

+(κ::Kernel, ψ::KernelSum) = KernelSum(ψ.a, κ, ψ.k...)
+(ψ::KernelSum, κ::Kernel) = KernelSum(ψ.a, ψ.k..., κ)

+(κ1::Kernel, κ2::Kernel) = KernelSum(1, κ1, κ2)
