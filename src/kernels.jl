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

for (kernel_object, kernel_op, kernel_array_op) in ((:KernelProduct, :*, :prod), (:KernelSum, :+, :sum))
    @eval begin

        immutable $kernel_object{T<:FloatingPoint} <: CompositeKernel{T}
            a::T
            k::Vector{Kernel{T}}
            function $kernel_object(a::T, κ::Vector{Kernel{T}})
                $(kernel_op == :+ ? :(>=) : :>)(a, 0) || error("a = $(a) must be greater than zero.")
                new(a, κ)
            end
        end
        $kernel_object{T<:FloatingPoint}(a::T, κ::Vector{Kernel{T}}) = $kernel_object{T}(a, κ)

        function $kernel_object(a::Real, κ::Kernel...)
            U = promote_type(typeof(a), map(eltype, κ)...)
            $kernel_object{U}(convert(U, a), Kernel{U}[κ...])
        end

        convert{T<:FloatingPoint}(::Type{$kernel_object{T}}, ψ::$kernel_object) = $kernel_object(convert(T, ψ.a), Kernel{T}[ψ.k...])
        convert{T<:FloatingPoint}(::Type{CompositeKernel{T}}, ψ::$kernel_object) = $kernel_object(convert(T, ψ.a), Kernel{T}[ψ.k...])
        convert{T<:FloatingPoint}(::Type{Kernel{T}}, ψ::$kernel_object) = $kernel_object(convert(T, ψ.a), Kernel{T}[ψ.k...])

        kernel{T<:FloatingPoint}(ψ::$kernel_object{T}, x::Vector{T}, y::Vector{T}) = $kernel_op(ψ.a, $kernel_array_op(map(κ -> kernel(κ,x,y), ψ.k)))
        kernel{T<:FloatingPoint}(ψ::$kernel_object{T}, x::T, y::T) = $kernel_op(ψ.a, $kernel_array_op(map(κ -> kernel(κ,x,y), ψ.k)))

        ismercer(ψ::$kernel_object) = all(ismercer, ψ.k)

        function description_string{T<:FloatingPoint}(ψ::$kernel_object{T}, eltype::Bool = true)
            descs = map(κ -> description_string(κ, false), ψ.k)
            if eltype
                $(string(kernel_object)) * (eltype ? "{$(T)}" : "") * "($(ψ.a), $(join(descs, ", ")))"
            else
                (ψ.a == 1 ? "" : "$(ψ.a)") * (length(descs) == 1 ? descs[1] : "($(join(descs, " " * $(string(kernel_op)) * " ")))")
            end
        end

        $kernel_op(a::Real, κ::Kernel) = $kernel_object(a, κ)
        $kernel_op(κ::Kernel, a::Real) = $kernel_op(a, κ)

        $kernel_op(a::Real, ψ::$kernel_object) = $kernel_object($kernel_op(a, ψ.a), ψ.k...)
        $kernel_op(ψ::$kernel_object, a::Real) = $kernel_op(a, ψ)

        $kernel_op(κ1::$kernel_object, κ2::$kernel_object) = $kernel_object($kernel_op(κ1.a, κ2.a), κ1.k..., κ2.k...)

        $kernel_op(κ::Kernel, ψ::$kernel_object) = $kernel_object(ψ.a, κ, ψ.k...)
        $kernel_op(ψ::$kernel_object, κ::Kernel) = $kernel_object(ψ.a, ψ.k..., κ)

        $kernel_op(κ1::Kernel, κ2::Kernel) = $kernel_object($(kernel_op == :+ ? 0 : 1), κ1, κ2)

    end
end
