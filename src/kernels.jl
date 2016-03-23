#===================================================================================================
  Kernels
===================================================================================================#

abstract Kernel{T<:AbstractFloat}

abstract StandardKernel{T<:AbstractFloat}  <: Kernel{T}  # Either a kernel is atomic or it is a
abstract KernelOperation{T<:AbstractFloat} <: Kernel{T}  # function of multiple kernels

eltype{T}(::Kernel{T}) = T 

doc"`ismercer(κ)`: tests whether a kernel `κ` is a Mercer kernel."
ismercer(::Kernel) = false

doc"`isnegdef(κ)`: tests whether a kernel `κ` is a continuous symmetric negative-definite kernel."
isnegdef(::Kernel) = false

attainszero(::Kernel)     = true
attainspositive(::Kernel) = true
attainsnegative(::Kernel) = true

isnonnegative(κ::Kernel) = !attainsnegative(κ)
ispositive(κ::Kernel)    = !attainsnegative(κ) && !attainszero(κ) &&  attainspositive(κ)
isnegative(k::Kernel)    =  attainsnegative(κ) && !attainszero(κ) && !attainspositive(κ)


#===================================================================================================
  Composition Classes: Valid kernel transformations
===================================================================================================#

abstract CompositionClass{T<:AbstractFloat}

eltype{T}(::CompositionClass{T}) = T 

iscomposable(::CompositionClass, ::Kernel) = false

attainszero(::CompositionClass)     = true
attainspositive(::CompositionClass) = true
attainsnegative(::CompositionClass) = true

include("definitions/compositionclasses.jl")  # standard CompositionClass definitions

function description_string(ϕ::CompositionClass)
    class = typeof(ϕ)
    fields = fieldnames(class)
    class_str = string(class.name.name)
    *(class_str, "(", join(["$field=$(getfield(ϕ,field).value)" for field in fields], ","), ")")
end

function show(io::IO, ϕ::CompositionClass)
    print(io, description_string(ϕ))
end

#=
for class_obj in concrete_subtypes(CompositionClass)
    class_name = class_obj.name.name  # symbol for concrete kernel type
    field_conversions = [:(convert(T, κ.$field.value)) for field in fieldnames(class_obj)]
    constructorcall = Expr(:call, class_name, field_conversions...)
    @eval begin
        convert{T<:AbstractFloat}(::Type{$class_name{T}}, κ::$class_name) = $constructorcall
    end
end
=#

#===================================================================================================
  Standard Kernels
===================================================================================================#

#==========================================================================
  Pairwise Kernels: consume two vectors
==========================================================================#

abstract PairwiseKernel{T<:AbstractFloat} <: StandardKernel{T}

abstract AdditiveKernel{T<:AbstractFloat} <: PairwiseKernel{T}

include("definitions/additivekernels.jl")

#=
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
=#

#==========================================================================
  Kernel Composition ψ = ϕ(κ(x,y))
==========================================================================#

doc"KernelComposition(ϕ,κ) = ϕ∘κ"
immutable KernelComposition{T<:AbstractFloat} <: StandardKernel{T}
    phi::CompositionClass{T}
    k::PairwiseKernel{T}
    function KernelComposition(ϕ::CompositionClass{T}, κ::PairwiseKernel{T})
        iscomposable(ϕ, κ) || error("Kernel is not composable.")
        new(ϕ, κ)
    end
end
function KernelComposition{T<:AbstractFloat}(ϕ::CompositionClass{T}, κ::PairwiseKernel{T})
    KernelComposition{T}(ϕ, κ)
end
#function KernelComposition{T<:AbstractFloat,U<:AbstractFloat}(
#        ϕ::CompositionClass{T}, 
#        κ::PairwiseKernel{U}
#    )
#    V = promote_type(T, U)
#    KernelComposition{V}(convert(CompositionClass{V}, ϕ), convert(Kernel{V}, κ))
#end

ismercer(κ::KernelComposition) = ismercer(κ.phi)
isnegdef(κ::KernelComposition) = isnegdef(κ.phi)

attainszero(κ::KernelComposition)     = attainszero(κ.phi)
attainspositive(κ::KernelComposition) = attainspositive(κ.phi)
attainsnegative(κ::KernelComposition) = attainsnegative(κ.phi)


#function convert{T<:AbstractFloat}(::Type{KernelComposition{T}}, ψ::KernelComposition)
#    KernelComposition(convert(CompositionClass{T}, ψ.phi), convert(Kernel{T}, ψ.k))
#end

# Special Compositions

∘(ϕ::CompositionClass, κ::Kernel) = KernelComposition(ϕ, κ)

function ^{T<:AbstractFloat}(κ::PairwiseKernel{T}, d::Integer)
    KernelComposition(PolynomialClass(one(T), zero(T), convert(T,d)), κ)
end

function ^{T<:AbstractFloat}(κ::PairwiseKernel{T}, γ::T)
    KernelComposition(PowerClass(one(T), zero(T), γ), κ)
end

function exp{T<:AbstractFloat}(κ::PairwiseKernel{T})
    KernelComposition(ExponentiatedClass(one(T), zero(T)), κ)
end

function tanh{T<:AbstractFloat}(κ::PairwiseKernel{T})
    KernelComposition(SigmoidClass(one(T), zero(T)), κ)
end


# Special Kernel Constructors

include("definitions/compositionkernels.jl")



#==========================================================================
  ARD Kernel
==========================================================================#

#=
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
=#



#===================================================================================================
  Kernel Operations
===================================================================================================#


#==========================================================================
  Kernel Affinity
==========================================================================#

doc"KernelAffinity(κ;a,c) = a⋅κ + c"
immutable KernelAffinity{T<:AbstractFloat} <: KernelOperation{T}
    a::Parameter{T}
    c::Parameter{T}
    k::Kernel{T}
    KernelAffinity(a::Variable{T}, c::Variable{T}, κ::Kernel{T}) = new(
        Parameter(a, LowerBound(zero(T), :strict)),
        Parameter(c, LowerBound(zero(T), :nonstrict)),
        κ
    )
end
function KernelAffinity{T<:AbstractFloat}(a::Argument{T}, c::Argument{T}, κ::Kernel{T})
    KernelAffinity{T}(convert(Variable{T}, a), convert(Variable{T}, c), κ)
end

ismercer(ψ::KernelAffinity) = ismercer(ψ.k)
isnegdef(ψ::KernelAffinity) = isnegdef(ψ.k)

attainszero(ψ::KernelAffinity)     = attainszero(ψ.k)
attainspositive(ψ::KernelAffinity) = attainspositive(ψ.k)
attainsnegative(ψ::KernelAffinity) = attainsnegative(ψ.k)


#function convert{T<:AbstractFloat}(::Type{KernelAffinity{T}}, ψ::KernelAffinity)
#    KernelAffinity(convert(T, ψ.a), convert(T, ψ.c), convert(Kernel{T}, ψ.k))
#end

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
    a::Parameter{T}
    k::Vector{Kernel{T}}
    function KernelProduct(a::Variable{T}, κ::Vector{Kernel{T}})
        if all(ismercer, κ)
            error("Kernels must be Mercer for closure under multiplication.")
        end
        new(Parameter(a, LowerBound(zero(T), :strict)), κ)
    end
end
#function KernelProduct{T<:AbstractFloat}(a::Argument{T}, κ::Vector{Kernel{T}})
#    KernelProduct{T}(convert(Variable{T}, a), κ)
#end

attainszero(ψ::KernelProduct)     = any(attainszero, ψ.k)
attainspositive(ψ::KernelProduct) = any(attainspositive, ψ.k)
attainsnegative(ψ::KernelProduct) = any(attainsnegative, ψ.k)

# Kernel Sum

immutable KernelSum{T<:AbstractFloat} <: KernelOperation{T}
    c::Parameter{T}
    k::Vector{Kernel{T}}
    function KernelSum(c::Variable{T}, κ::Vector{Kernel{T}})
        if !(all(ismercer, κ) || all(isnegdef, κ))
            error("All kernels must be Mercer or negative definite for closure under addition")
        end
        new(Parameter(c, LowerBound(zero(T), :nonstrict)), κ)
    end
end
#function KernelSum{T<:AbstractFloat}(c::Argument{T}, κ::Vector{Kernel{T}})
#    KernelSum{T}(convert(Variable{T}, c), κ)
#end

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
        
        function $kernel_object{T<:AbstractFloat}($scalar::Argument{T}, κ::Vector{Kernel{T}})
            ($kernel_object){T}(convert(Variable{T}, $scalar), κ)
        end

        function $kernel_object{T<:AbstractFloat}($scalar::Argument{T}, κ::Kernel{T}...)
            ($kernel_object){T}($scalar, Kernel{T}[κ...])
        end
        
        function $kernel_object{T<:AbstractFloat}($scalar::Argument, κ::Kernel{T}...)
            ($kernel_object){T}(convert(Argument{T}, $scalar), Kernel{T}[κ...])
        end

        #function description_string{T<:AbstractFloat}(ψ::$kernel_object{T}, eltype::Bool = true)
        #    descs = map(κ -> description_string(κ, false), ψ.k)
        #    ($(string(kernel_object)) * (eltype ? "{$(T)}" : "") * 
        #        "(" * (ψ.$scalar == $identity ? "" : $(string(scalar)) * "=" * string(ψ.$scalar) * 
        #        ",") * join(descs, ", ") * ")")
        #end

        #function convert{T<:AbstractFloat}(::Type{($kernel_object){T}}, ψ::$kernel_object)
        #    $kernel_object(convert(T, ψ.$scalar), Kernel{T}[ψ.k...])
        #end

        ismercer(ψ::$kernel_object) = all(ismercer, ψ.k)
        isnegdef(ψ::$kernel_object) = all(isnegdef, ψ.k)

        #=

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
        =#

    end
end

#=
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
=#


#===================================================================================================
  Conversions
===================================================================================================#

#=
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
=#
