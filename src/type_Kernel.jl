#===================================================================================================
  Kernels
===================================================================================================#

abstract Kernel{T<:AbstractFloat}

eltype{T}(::Kernel{T}) = T 

doc"`ismercer(κ)`: returns `true` if kernel `κ` is a Mercer kernel."
ismercer(::Kernel) = false

doc"`isnegdef(κ)`: returns `true` if kernel `κ` is a continuous symmetric negative-definite kernel."
isnegdef(::Kernel) = false

doc"`attainszero(κ)`: returns `true` if ∃x,y such that κ(x,y) = 0"
attainszero(::Kernel) = true

doc"`attainspositive(κ)`: returns `true` if ∃x,y such that κ(x,y) > 0"
attainspositive(::Kernel) = true

doc"`attainsnegative(κ)`: returns `true` if ∃x,y such that κ(x,y) < 0"
attainsnegative(::Kernel) = true

doc"`isnonnegative(κ)`: returns `true` if κ(x,y) > 0 ∀x,y"
isnonnegative(κ::Kernel) = !attainsnegative(κ)

doc"`ispositive(κ)`: returns `true` if κ(x,y) ≧ 0 ∀x,y"
ispositive(κ::Kernel) = !attainsnegative(κ) && !attainszero(κ) &&  attainspositive(κ)

doc"`isnegative(κ)`: returns `true` if κ(x,y) ≦ 0 ∀x,y"
isnegative(k::Kernel) =  attainsnegative(κ) && !attainszero(κ) && !attainspositive(κ)

function show(io::IO, κ::Kernel)
    print(io, description_string(κ))
end

function convert{T<:AbstractFloat,K<:Kernel}(::Type{Kernel{T}}, ϕ::K)
    convert(K.name.primary{T}, ϕ)
end


#== Standard Kernels ==#

abstract StandardKernel{T<:AbstractFloat}  <: Kernel{T}  # Either a kernel is atomic or it is a

include("type_PairwiseKernel.jl")
include("type_CompositionClass.jl")
include("type_KernelComposition.jl")


#== Kernel Operations ==#

abstract KernelOperation{T<:AbstractFloat} <: Kernel{T}  # function of multiple kernels

#==========================================================================
  Kernel Affinity
==========================================================================#

doc"KernelAffinity(κ;a,c) = a⋅κ + c"
immutable KernelAffinity{T<:AbstractFloat} <: KernelOperation{T}
    a::Parameter{T}
    c::Parameter{T}
    kappa::Kernel{T}
    KernelAffinity(a::Variable{T}, c::Variable{T}, κ::Kernel{T}) = new(
        Parameter(a, LowerBound(zero(T), :strict)),
        Parameter(c, LowerBound(zero(T), :nonstrict)),
        κ
    )
end
function KernelAffinity{T<:AbstractFloat}(a::Argument{T}, c::Argument{T}, κ::Kernel{T})
    KernelAffinity{T}(convert(Variable{T}, a), convert(Variable{T}, c), κ)
end

ismercer(ψ::KernelAffinity) = ismercer(ψ.kappa)
isnegdef(ψ::KernelAffinity) = isnegdef(ψ.kappa)

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
