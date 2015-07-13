#===================================================================================================
  Generic Kernels
===================================================================================================#

typealias KernelInput{T} Union(T,Vector{T})
abstract Kernel{T<:FloatingPoint}

eltype{T}(κ::Kernel{T}) = T

#call{T<:FloatingPoint}(κ::Kernel{T}, x::Vector{T}, y::Vector{T}) = kernel(κ, x, y)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}) = kernelmatrix(κ, X)
#call{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}) = kernelmatrix(κ, X, Y)

ismercer(::Kernel) = false
iscondposdef(κ::Kernel) = ismercer(κ)

function show(io::IO, κ::Kernel)
    print(io, description_string(κ))
end

abstract SimpleKernel{T<:FloatingPoint} <: Kernel{T}
abstract CompositeKernel{T<:FloatingPoint} <: Kernel{T}


#===================================================================================================
  Simple Kernels
===================================================================================================#

abstract StandardKernel{T<:FloatingPoint} <: SimpleKernel{T}


#===========================================================================
  Scalar Product Kernels - kernels of the form k(x,y) = κ(xᵀy)
===========================================================================#

abstract ScalarProductKernel{T<:FloatingPoint} <: StandardKernel{T}

kernel{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::KernelInput{T}, y::KernelInput{T}) = kappa(κ, scprod(x, y))

# Scalar Product Kernel definitions
include("standardkernels/scalarproduct.jl")


#===========================================================================
  Squared Distance Kernels - kernels of the form k(x,y) = κ((x-y)ᵀ(x-y))
===========================================================================#

abstract SquaredDistanceKernel{T<:FloatingPoint} <: StandardKernel{T}

kernel{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::KernelInput{T}, y::KernelInput{T}) = kappa(κ, sqdist(x, y))

# Squared Distance Kernel definitions
include("standardkernels/squareddistance.jl")


#===========================================================================
  Automatic Relevance Determination (ARD) kernels
===========================================================================#

typealias ARDKernelTypes{T<:FloatingPoint} Union(SquaredDistanceKernel{T}, ScalarProductKernel{T})

immutable ARD{T<:FloatingPoint,K<:StandardKernel{T}} <: SimpleKernel{T}
    k::K
    w::Vector{T}
    function ARD(κ::K, w::Vector{T})
        all(w .>= 0) || throw(ArgumentError("All elements of w = $(w) must be non-negative."))
        new(κ, copy(w))
    end
end

ARD{T<:FloatingPoint}(κ::ARDKernelTypes{T}, w::Vector{T}) = ARD{T,typeof(κ)}(κ, w)
ARD{T<:FloatingPoint}(κ::ARDKernelTypes{T}, dim::Integer) = ARD{T,typeof(κ)}(κ, ones(T, dim))

function description_string{T<:FloatingPoint,K<:StandardKernel}(κ::ARD{T,K}, eltype::Bool = true)
    "ARD" * (eltype ? "{$(T)}" : "") * "(κ=$(description_string(κ.k, false)), w=$(κ.w))"
end

kernel{T<:FloatingPoint,U<:StandardKernel}(κ::ARD{T,U}, x::Vector{T}, y::Vector{T}) = kernel(κ.k, x.*κ.w, y.*κ.w)  # Default scaling
kernel{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Vector{T}, y::Vector{T}) = kappa(κ.k, sqdist(x, y, κ.w))
kernel{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Vector{T}, y::Vector{T}) = kappa(κ.k, scprod(x, y, κ.w))

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
    kernelobjectsym = kernelobject.name.name  # symbol for concrete kernel type

    fieldconversions = [:(convert(T, κ.$field)) for field in names(kernelobject)]
    constructorcall = Expr(:call, kernelobjectsym, fieldconversions...)

    @eval begin
        convert{T<:FloatingPoint}(::Type{$kernelobjectsym{T}}, κ::$kernelobjectsym) = $constructorcall
    end
end

function convert{T<:FloatingPoint}(::Type{ARD{T}}, κ::ARD)
    ARD(convert(Kernel{T}, κ.k), T[κ.w...])
end

for kernelobject in concretesubtypes(Kernel)
    kernelobjectsym = kernelobject.name.name  # symbol for concrete kernel type

    for kerneltype in supertypes(kernelobject)
        kerneltypesym = kerneltype.name.name  # symbol for abstract supertype

        @eval begin
            function convert{T<:FloatingPoint}(::Type{$kerneltypesym{T}}, κ::$kernelobjectsym)
                convert($kernelobjectsym{T}, κ)
            end
        end
    end
end


#===================================================================================================
  Composite Kernels
===================================================================================================#

for (kernel_object, kernel_op, kernel_array_op, identity) in (
        (:KernelProduct, :*, :prod, :1),
        (:KernelSum,     :+, :sum,  :0)
    )
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

        kernel{T<:FloatingPoint}(ψ::$kernel_object{T}, x::KernelInput{T}, y::KernelInput{T}) = $kernel_op(ψ.a, $kernel_array_op(map(κ -> kernel(κ,x,y), ψ.k)))

        ismercer(ψ::$kernel_object) = all(ismercer, ψ.k)
        iscondposdef(ψ::$kernel_object) = all(iscondposdef, ψ.k)

        function description_string{T<:FloatingPoint}(ψ::$kernel_object{T}, eltype::Bool = true)
            descs = map(κ -> description_string(κ, false), ψ.k)
            if eltype
                $(string(kernel_object)) * (eltype ? "{$(T)}" : "") * "($(ψ.a), $(join(descs, ", ")))"
            else
                if $kernel_op !== (*) && ψ.a != $identity
                    insert!(descs, 1, string(ψ.a))
                end
                desc_str = string("(", join(descs, $(string(" ", kernel_op, " "))), ")")
                if $kernel_op === (*) && ψ.a != $identity
                    string(ψ.a, desc_str)
                else
                    desc_str
                end
            end
        end

        $kernel_op(a::Real, κ::Kernel) = $kernel_object(a, κ)
        $kernel_op(κ::Kernel, a::Real) = $kernel_op(a, κ)

        $kernel_op(a::Real, ψ::$kernel_object) = $kernel_object($kernel_op(a, ψ.a), ψ.k...)
        $kernel_op(ψ::$kernel_object, a::Real) = $kernel_op(a, ψ)

        $kernel_op(κ1::$kernel_object, κ2::$kernel_object) = $kernel_object($kernel_op(κ1.a, κ2.a), κ1.k..., κ2.k...)

        $kernel_op(κ::Kernel, ψ::$kernel_object) = $kernel_object(ψ.a, κ, ψ.k...)
        $kernel_op(ψ::$kernel_object, κ::Kernel) = $kernel_object(ψ.a, ψ.k..., κ)

        $kernel_op(κ1::Kernel, κ2::Kernel) = $kernel_object($identity, κ1, κ2)

    end
end
