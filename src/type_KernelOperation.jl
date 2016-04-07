#===================================================================================================
  Kernel Affinity
===================================================================================================#

doc"KernelAffinity(κ;a,c) = a⋅κ + c"
immutable KernelAffinity{T<:AbstractFloat} <: KernelOperation{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    kappa::Kernel{T}
    KernelAffinity(a::Variable{T}, c::Variable{T}, κ::Kernel{T}) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed)),
        κ
    )
end
function KernelAffinity{T<:AbstractFloat}(a::Argument{T}, c::Argument{T}, κ::Kernel{T})
    KernelAffinity{T}(Variable(a), Variable(c), κ)
end

ismercer(ψ::KernelAffinity) = ismercer(ψ.kappa)
isnegdef(ψ::KernelAffinity) = isnegdef(ψ.kappa)

attainszero(ψ::KernelAffinity)     = attainszero(ψ.kappa)
attainspositive(ψ::KernelAffinity) = attainspositive(ψ.kappa)
attainsnegative(ψ::KernelAffinity) = attainsnegative(ψ.kappa)

function description_string(κ::KernelAffinity)
    "KernelAffinity(a=$(κ.a.value),c=$(κ.c.value)," * description_string(κ.kappa) * ")"
end

function convert{T<:AbstractFloat}(::Type{KernelAffinity{T}}, ψ::KernelAffinity)
    KernelAffinity(convert(T, ψ.a.value), convert(T, ψ.c.value), convert(Kernel{T}, ψ.kappa))
end

@inline phi{T<:AbstractFloat}(ψ::KernelAffinity{T}, z::T) = ψ.a*z + ψ.c


# Operations

+{T<:AbstractFloat}(κ::Kernel{T}, c::Real) = KernelAffinity(one(T), convert(T, c), κ)
+(c::Real, κ::Kernel) = +(κ, c)

*{T<:AbstractFloat}(κ::Kernel{T}, a::Real) = KernelAffinity(convert(T, a), zero(T), κ)
*(a::Real, κ::Kernel) = *(κ, a)

function +{T<:AbstractFloat}(κ::KernelAffinity{T}, c::Real)
    KernelAffinity(κ.a.value, κ.c + convert(T,c), κ.kappa)
end
+(c::Real, κ::KernelAffinity) = +(κ, c)

function *{T<:AbstractFloat}(κ::KernelAffinity{T}, a::Real)
    a = convert(T, a)
    KernelAffinity(a * κ.a, a * κ.c, κ.kappa)
end
*(a::Real, κ::KernelAffinity) = *(κ, a)

function ^{T<:AbstractFloat}(ψ::KernelAffinity{T}, d::Integer)
    KernelComposition(PolynomialClass(ψ.a.value, ψ.c.value, convert(T,d)), ψ.kappa)
end

function ^{T<:AbstractFloat}(ψ::KernelAffinity{T}, γ::AbstractFloat)
    KernelComposition(PowerClass(ψ.a.value, ψ.c.value, convert(T,γ)), ψ.kappa)
end

function exp{T<:AbstractFloat}(ψ::KernelAffinity{T})
    KernelComposition(ExponentiatedClass(ψ.a.value, ψ.c.value), ψ.kappa)
end

function tanh{T<:AbstractFloat}(ψ::KernelAffinity{T})
    KernelComposition(SigmoidClass(ψ.a.value, ψ.c.value), ψ.kappa)
end


#===================================================================================================
  Kernel Product and Sum
===================================================================================================#

# Kernel Product

immutable KernelProduct{T<:AbstractFloat} <: KernelOperation{T}
    a::HyperParameter{T}
    kappa1::Kernel{T}
    kappa2::Kernel{T}
    function KernelProduct(a::Variable{T}, κ1::Kernel{T}, κ2::Kernel)
        if !(ismercer(κ1) && ismercer(κ2))
            error("Kernels must be Mercer for closure under multiplication.")
        end
        new(HyperParameter(a, leftbounded(zero(T), :open)), κ1, κ2)
    end
end
function KernelProduct{T<:AbstractFloat}(a::Argument{T}, κ1::Kernel{T}, κ2::Kernel{T})
    KernelProduct{T}(Variable(a), κ1, κ2)
end


# Kernel Sum

immutable KernelSum{T<:AbstractFloat} <: KernelOperation{T}
    c::HyperParameter{T}
    kappa1::Kernel{T}
    kappa2::Kernel{T}
    function KernelSum(c::Variable{T}, κ1::Kernel{T}, κ2::Kernel{T})
        if !(ismercer(κ1) && ismercer(κ2)) && !(isnegdef(κ1) && isnegdef(κ2))
            error("All kernels must be Mercer or negative definite for closure under addition")
        end
        new(HyperParameter(c, leftbounded(zero(T), :closed)), κ1, κ2)
    end
end
function KernelSum{T<:AbstractFloat}(c::Argument{T}, κ1::Kernel{T}, κ2::Kernel{T})
    KernelSum{T}(Variable(c), κ1, κ2)
end


# Common Functions

for (kernel_object, kernel_op, identity, scalar) in (
        (:KernelProduct, :*, :1, :a),
        (:KernelSum,     :+, :0, :c)
    )
    other_identity = identity == :1 ? :0 : :1
    scalar_str = string(scalar)
    @eval begin
        
        function description_string(ψ::$kernel_object, showtype::Bool = true)
            constant_str = string($scalar_str,"=", ψ.$scalar.value)
            kernel1_str = "kappa1=" * description_string(ψ.kappa1, false)
            kernel2_str = "kappa2=" * description_string(ψ.kappa2, false)
            obj_str = string($kernel_object.name.name, showtype ? string("{", eltype(ψ), "}") : "")
            string(obj_str, "(", constant_str, ",", kernel1_str, ",", kernel2_str, ")")
        end

        function convert{T<:AbstractFloat}(::Type{($kernel_object){T}}, ψ::$kernel_object)
            $kernel_object(Variable(convert(T, ψ.$scalar.value), ψ.$scalar.isfixed),
                           convert(Kernel{T}, ψ.kappa1), convert(Kernel{T}, ψ.kappa2))
        end

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
