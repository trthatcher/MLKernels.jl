#===================================================================================================
  Kernels & Composition Classes
===================================================================================================#

doc"
`Kernel`: `κ` such that `κ` is a Mercer kernel or a continuous symmetric 
negative-definite kernel.
"
abstract Kernel{T<:AbstractFloat}

doc"`CompositionClass`: `ϕ` such that `ϕ(κ(x,y))` is a kernel for some kernel `κ`."
abstract CompositionClass{T<:AbstractFloat}

function show(io::IO, κ::Union{Kernel,CompositionClass})
    print(io, description_string(κ))
end

eltype{T}(::Union{Kernel{T},CompositionClass{T}}) = T

doc"`ismercer(κ)`: tests whether a kernel `κ` is a Mercer kernel."
ismercer(::Union{Kernel,CompositionClass}) = false

doc"`isnegdef(κ)`: tests whether a kernel `κ` is a continuous symmetric negative-definite kernel."
isnegdef(::Union{Kernel,CompositionClass}) = false

attainszero(::Union{Kernel,CompositionClass})     = true
attainspositive(::Union{Kernel,CompositionClass}) = true
attainsnegative(::Union{Kernel,CompositionClass}) = true

isnonnegative(κ::Union{Kernel,CompositionClass}) = !attainsnegative(κ)
function ispositive(κ::Union{Kernel,CompositionClass})
    !attainsnegative(κ) && !attainszero(κ) && attainspositive(κ)
end
function isnegative(k::Union{Kernel,CompositionClass})
    attainsnegative(κ) && !attainszero(κ) && !attainspositive(κ)
end


#==========================================================================
Composition Classes
==========================================================================#

iscomposable(::CompositionClass, ::Kernel) = true

include("definitions/compositionclasses.jl")

for class_obj in concrete_subtypes(CompositionClass)
  class_name = class_obj.name.name  # symbol for concrete kernel type
  field_conversions = [:(convert(T, κ.$field)) for field in fieldnames(class_obj)]
  constructorcall = Expr(:call, class_name, field_conversions...)
  @eval begin
      convert{T<:AbstractFloat}(::Type{$class_name{T}}, κ::$class_name) = $constructorcall
  end
end


#===================================================================================================
  Standard Kernels
===================================================================================================#

abstract StandardKernel{T<:AbstractFloat} <: Kernel{T}
abstract BaseKernel{T<:AbstractFloat} <: StandardKernel{T}


#==========================================================================
  Additive Kernel: k(x,y) = sum(k(x_i,y_i))    x ∈ ℝⁿ, y ∈ ℝⁿ
==========================================================================#

abstract AdditiveKernel{T<:AbstractFloat} <: BaseKernel{T}

include("definitions/additivekernels.jl")

for kernel_obj in concrete_subtypes(AdditiveKernel)
    kernel_name = kernel_obj.name.name  # symbol for concrete kernel type
    if length(fieldnames(kernel_obj)) == 0
        @eval begin
            convert{T<:AbstractFloat}(::Type{$kernel_name{T}}, κ::$kernel_name) = $kernel_name{T}()
        end
    else 
        field_conversions = [:(convert(T, κ.$field)) for field in fieldnames(kernel_obj)]
        constructor_call = Expr(:call, kernel_name, field_conversions...)
        @eval begin
            convert{T<:AbstractFloat}(::Type{$kernel_name{T}}, κ::$kernel_name) = $constructor_call
        end
    end
end


#==========================================================================
  ARD Kernel
==========================================================================#

doc"`ARD(κ,w)` where `κ <: AdditiveKernel`"
immutable ARD{T<:AbstractFloat} <: BaseKernel{T}
    k::AdditiveKernel{T}
    w::Vector{T}
    function ARD(κ::AdditiveKernel{T}, w::Vector{T})
        length(w) > 0 || error("Weight vector w must be at least of length 1.")
        all(w .> 0) || error("Weight vector w must consist of positive values.")
        new(κ, w)
    end
end
ARD{T<:AbstractFloat}(κ::AdditiveKernel{T}, w::Vector{T}) = ARD{T}(κ, w)

ismercer(κ::ARD) = ismercer(κ.k)
isnegdef(κ::ARD) = isnegdef(κ.k)

attainszero(κ::ARD) = attainszero(κ.k)
attainspositive(κ::ARD) = attainspositive(κ.k)
attainsnegative(κ::ARD) = attainsnegative(κ.k)

function description_string{T<:AbstractFloat,}(κ::ARD{T}, eltype::Bool = true)
    "ARD" * (eltype ? "{$(T)}" : "") * "(κ=$(description_string(κ.k, false)),w=$(κ.w))"
end

function convert{T<:AbstractFloat}(::Type{ARD{T}}, κ::ARD)
    ARD(convert(Kernel{T},κ.k), convert(Vector{T}, κ.w))
end


#==========================================================================
  Kernel Composition ψ = ϕ(κ(x,y))
==========================================================================#

doc"KernelComposition(ϕ,κ) = ϕ∘κ"
immutable KernelComposition{T<:AbstractFloat} <: StandardKernel{T}
    phi::CompositionClass{T}
    k::Kernel{T}
    function KernelComposition(ϕ::CompositionClass{T}, κ::Kernel{T})
        iscomposable(ϕ, κ) || error("Kernel is not composable.")
        new(ϕ, κ)
    end
end
function KernelComposition{T<:AbstractFloat}(ϕ::CompositionClass{T}, κ::Kernel{T})
    KernelComposition{T}(ϕ, κ)
end

ismercer(κ::KernelComposition) = ismercer(κ.phi)
isnegdef(κ::KernelComposition) = isnegdef(κ.phi)

attainszero(κ::KernelComposition) = attainszero(κ.phi)
attainspositive(κ::KernelComposition) = attainspositive(κ.phi)
attainsnegative(κ::KernelComposition) = attainsnegative(κ.phi)

function description_string{T<:AbstractFloat,}(κ::KernelComposition{T}, eltype::Bool = true)
    "KernelComposition" * (eltype ? "{$(T)}" : "") * "(ϕ=$(description_string(κ.phi,false))," *
    "κ=$(description_string(κ.k, false)))"
end

function convert{T<:AbstractFloat}(::Type{KernelComposition{T}}, ψ::KernelComposition)
    KernelComposition(convert(CompositionClass{T}, ψ.phi), convert(Kernel{T}, ψ.k))
end

# Special Compositions

∘(ϕ::CompositionClass, κ::Kernel) = KernelComposition(ϕ, κ)

function ^{T<:AbstractFloat}(κ::Kernel{T}, d::Integer)
    KernelComposition(PolynomialClass(one(T), zero(T), convert(T,d)), κ)
end

function ^{T<:AbstractFloat}(κ::Kernel{T}, γ::T)
    KernelComposition(PowerClass(one(T), zero(T), γ), κ)
end

function exp{T<:AbstractFloat}(κ::Kernel{T})
    KernelComposition(ExponentiatedClass(one(T), zero(T)), κ)
end

function tanh{T<:AbstractFloat}(κ::Kernel{T})
    KernelComposition(SigmoidClass(one(T), zero(T)), κ)
end


# Special Kernel Constructors

include("definitions/compositionkernels.jl")


#===================================================================================================
  Kernel Operations
===================================================================================================#

abstract KernelOperation{T<:AbstractFloat} <: Kernel{T}

#==========================================================================
  Kernel Affinity
==========================================================================#

doc"KernelAffinity(κ;a,c) = a⋅κ + c"
immutable KernelAffinity{T<:AbstractFloat} <: KernelOperation{T}
    a::T
    c::T
    k::Kernel{T}
    function KernelAffinity(a::T, c::T, κ::Kernel{T})
        a > 0 || error("a = $(a) must be greater than zero.")
        c >= 0 || error("c = $(c) must be greater than or equal to zero.")
        new(a, c, κ)
    end
end
KernelAffinity{T<:AbstractFloat}(a::T, c::T, κ::Kernel{T}) = KernelAffinity{T}(a, c, κ)

ismercer(ψ::KernelAffinity) = ismercer(ψ.k)
isnegdef(ψ::KernelAffinity) = isnegdef(ψ.k)

attainszero(ψ::KernelAffinity) = attainszero(ψ.k)
attainspositive(ψ::KernelAffinity) = attainspositive(ψ.k)
attainsnegative(ψ::KernelAffinity) = attainsnegative(ψ.k)

function description_string{T<:AbstractFloat}(ψ::KernelAffinity{T}, eltype::Bool = true)
    "Affine" * (eltype ? "{$(T)}" : "") * "(a=$(ψ.a),c=$(ψ.c),κ=" * 
    description_string(ψ.k) * ")"
end

function convert{T<:AbstractFloat}(::Type{KernelAffinity{T}}, ψ::KernelAffinity)
    KernelAffinity(convert(T, ψ.a), convert(T, ψ.c), convert(Kernel{T}, ψ.k))
end

@inline phi{T<:AbstractFloat}(ψ::KernelAffinity{T}, z::T) = ψ.a*z + ψ.c

# Operations

+{T<:AbstractFloat}(κ::Kernel{T}, c::Real) = KernelAffinity(one(T), convert(T, c), κ)
+(c::Real, κ::Kernel) = +(κ, c)

*{T<:AbstractFloat}(κ::Kernel{T}, a::Real) = KernelAffinity(convert(T, a), zero(T), κ)
*(a::Real, κ::Kernel) = *(κ, a)

+{T<:AbstractFloat}(κ::KernelAffinity{T}, c::Real) = KernelAffinity(κ.a, κ.c + convert(T, c), κ.k)
+(c::Real, κ::KernelAffinity) = +(κ, c)

function *{T<:AbstractFloat}(κ::KernelAffinity{T}, a::Real)
    a = convert(T, a)
    KernelAffinity(a * κ.a, a * κ.c, κ.k)
end
*(a::Real, κ::KernelAffinity) = *(κ, a)

function ^{T<:AbstractFloat}(ψ::KernelAffinity{T}, d::Integer)
    KernelComposition(PolynomialClass(ψ.a, ψ.c, convert(T,d)), ψ.k)
end

function ^{T<:AbstractFloat}(ψ::KernelAffinity{T}, γ::AbstractFloat)
    KernelComposition(PowerClass(ψ.a, ψ.c, convert(T,γ)), ψ.k)
end

function exp{T<:AbstractFloat}(ψ::KernelAffinity{T})
    KernelComposition(ExponentiatedClass(ψ.a, ψ.c), ψ.k)
end

function tanh{T<:AbstractFloat}(ψ::KernelAffinity{T})
    KernelComposition(SigmoidClass(ψ.a, ψ.c), ψ.k)
end


#==========================================================================
  Kernel Product & Kernel Sum
==========================================================================#

# Kernel Product

immutable KernelProduct{T<:AbstractFloat} <: KernelOperation{T}
    a::T
    k::Vector{Kernel{T}}
    function KernelProduct(a::T, κ::Vector{Kernel{T}})
        a > 0 || error("a = $(a) must be greater than zero.")
        if length(κ) > 1
            all(ismercer, κ) || error("Kernels must be Mercer for closure under multiplication.")
        end
        new(a, κ)
    end
end

attainszero(ψ::KernelProduct) = any(attainszero, ψ.k)
attainspositive(ψ::KernelProduct) = any(attainspositive, ψ.k)
attainsnegative(ψ::KernelProduct) = any(attainsnegative, ψ.k)

# Kernel Sum

immutable KernelSum{T<:AbstractFloat} <: KernelOperation{T}
    c::T
    k::Vector{Kernel{T}}
    function KernelSum(c::T, κ::Vector{Kernel{T}})
        c >= 0 || error("c = $(c) must be greater than or equal to zero.")
        if !(all(ismercer, κ) || all(isnegdef, κ))
            error("All kernels must be Mercer or negative definite for closure under addition")
        end
        new(c, κ)
    end
end

attainszero(ψ::KernelSum) = (all(attainszero, ψ.k) && ψ.c == 0) || (any(attainspositive, ψ.k) &&
                                                                    any(attainsnegative, ψ.k))
attainspositive(ψ::KernelSum) = any(attainspositive, ψ.k)
attainsnegative(ψ::KernelSum) = any(attainsnegative, ψ.k)


# Common Functions

for (kernel_object, kernel_op, identity, scalar) in (
        (:KernelProduct, :*, :1, :a),
        (:KernelSum,     :+, :0, :c)
    )
    other_identity = identity == :1 ? :0 : :1
    @eval begin
        
        function $kernel_object{T<:AbstractFloat}($scalar::T, κ::Vector{Kernel{T}})
            ($kernel_object){T}($scalar, κ)
        end

        function $kernel_object{T<:AbstractFloat}($scalar::T, κ::Kernel{T}...)
            ($kernel_object){T}($scalar, Kernel{T}[κ...])
        end
        
        function $kernel_object{T<:AbstractFloat}($scalar::Real, κ::Kernel{T}...)
            ($kernel_object){T}(convert(T, $scalar), Kernel{T}[κ...])
        end

        function description_string{T<:AbstractFloat}(ψ::$kernel_object{T}, eltype::Bool = true)
            descs = map(κ -> description_string(κ, false), ψ.k)
            ($(string(kernel_object)) * (eltype ? "{$(T)}" : "") * 
                "(" * (ψ.$scalar == $identity ? "" : $(string(scalar)) * "=" * string(ψ.$scalar) * 
                ",") * join(descs, ", ") * ")")
        end

        function convert{T<:AbstractFloat}(::Type{($kernel_object){T}}, ψ::$kernel_object)
            $kernel_object(convert(T, ψ.$scalar), Kernel{T}[ψ.k...])
        end

        ismercer(ψ::$kernel_object) = all(ismercer, ψ.k)
        isnegdef(ψ::$kernel_object) = all(isnegdef, ψ.k)

        function $kernel_op($scalar::Real, ψ::$kernel_object) 
            $kernel_object($kernel_op($scalar, ψ.$scalar), ψ.k...)
        end
        $kernel_op(ψ::$kernel_object, $scalar::Real) = $kernel_op($scalar, ψ)

        function $kernel_op(κ1::$kernel_object, κ2::$kernel_object)
            $kernel_object($kernel_op(κ1.$scalar, κ2.$scalar), κ1.k..., κ2.k...)
        end

        $kernel_op(κ::Kernel, ψ::$kernel_object) = $kernel_object(ψ.$scalar, κ, ψ.k...)
        $kernel_op(ψ::$kernel_object, κ::Kernel) = $kernel_object(ψ.$scalar, ψ.k..., κ)

        $kernel_op(κ1::Kernel, κ2::Kernel) = $kernel_object($identity, κ1, κ2)

    end
end

for (kernel_object, kernel_op, identity, scalar, op2_identity, op2_scalar) in (
        (:KernelProduct, :*, :1, :a, :0, :c),
        (:KernelSum,     :+, :0, :c, :1, :a)
    )
    @eval begin
        function $kernel_op(κ1::KernelAffinity, κ2::KernelAffinity)
            if κ1.$op2_scalar == $op2_identity && κ2.$op2_scalar == $op2_identity
                $kernel_object($kernel_op(κ1.$scalar, κ2.$scalar), κ1.k, κ2.k)
            else
                $kernel_object($identity, κ1, κ2)
            end
        end

        function $kernel_op(κ1::KernelAffinity, κ2::StandardKernel)
            if κ1.$op2_scalar == $op2_identity
                $kernel_object(κ1.$scalar, κ1.k, κ2)
            else
                $kernel_object($identity, κ1, κ2)
            end
        end
        $kernel_op(κ1::StandardKernel, κ2::KernelAffinity) = $kernel_op(κ2, κ1)
    end
end



#===================================================================================================
  Conversions
===================================================================================================#

for kernel in (
        concrete_subtypes(AdditiveKernel)..., 
        ARD, 
        concrete_subtypes(CompositionClass)...,
        KernelComposition,
        KernelAffinity,
        KernelSum,
        KernelProduct
    )
    kernel_sym = kernel.name.name  # symbol for concrete kernel type
    for parent in supertypes(kernel)
        parent_sym = parent.name.name  # symbol for abstract supertype
        @eval begin
            function convert{T<:AbstractFloat}(::Type{$parent_sym{T}}, κ::$kernel_sym)
                convert($kernel_sym{T}, κ)
            end
        end
    end
end
