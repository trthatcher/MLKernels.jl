#===================================================================================================
  RealKernel Affinity
===================================================================================================#

abstract PointwiseKernel{T} <: RealKernel{T}

doc"AffineKernel(f;a,c) = a⋅f + c"
immutable AffineKernel{T<:AbstractFloat} <: PointwiseKernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    f::RealKernel{T}
    AffineKernel(a::Variable{T}, c::Variable{T}, f::RealKernel{T}) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed)),
        f
    )
end
function AffineKernel{T<:AbstractFloat}(a::Argument{T}, c::Argument{T}, f::RealKernel{T})
    AffineKernel{T}(Variable(a), Variable(c), f)
end

function ==(f::AffineKernel, g::AffineKernel)
    (f.a == g.a) && (f.c == g.c) && (f.f == g.f)
end

ismercer(h::AffineKernel) = ismercer(h.f)
isnegdef(h::AffineKernel) = isnegdef(h.f)

attainszero(h::AffineKernel)     = attainszero(h.f)
attainspositive(h::AffineKernel) = attainspositive(h.f)
attainsnegative(h::AffineKernel) = attainsnegative(h.f)

function description_string(h::AffineKernel, showtype::Bool = true)
    obj_str = string("AffineKernel", showtype ? string("{", eltype(h), "}") : "")
    h_str = description_string(h.f)
    string(obj_str, "(a=$(h.a.value),c=$(h.c.value),", h_str, ")")
end

function convert{T<:AbstractFloat}(::Type{AffineKernel{T}}, h::AffineKernel)
    AffineKernel(convert(T, h.a.value), convert(T, h.c.value), convert(RealKernel{T}, h.f))
end


# Operations

+{T<:AbstractFloat}(f::RealKernel{T}, c::Real) = AffineKernel(one(T), convert(T, c), f)
+(c::Real, f::RealKernel) = +(f, c)

*{T<:AbstractFloat}(f::RealKernel{T}, a::Real) = AffineKernel(convert(T, a), zero(T), f)
*(a::Real, f::RealKernel) = *(f, a)

function +{T<:AbstractFloat}(f::AffineKernel{T}, c::Real)
    AffineKernel(f.a.value, f.c + convert(T,c), f.f)
end
+(c::Real, f::AffineKernel) = +(f, c)

function *{T<:AbstractFloat}(f::AffineKernel{T}, a::Real)
    a = convert(T, a)
    AffineKernel(a * f.a, a * f.c, f.f)
end
*(a::Real, f::AffineKernel) = *(f, a)

function ^{T<:AbstractFloat}(h::AffineKernel{T}, d::Integer)
    CompositeKernel(PolynomialClass(h.a.value, h.c.value, d), h.f)
end

function ^{T<:AbstractFloat}(h::AffineKernel{T}, γ::AbstractFloat)
    CompositeKernel(PowerClass(h.a.value, h.c.value, convert(T,γ)), h.f)
end

function exp{T<:AbstractFloat}(h::AffineKernel{T})
    CompositeKernel(ExponentiatedClass(h.a.value, h.c.value), h.f)
end

function tanh{T<:AbstractFloat}(h::AffineKernel{T})
    CompositeKernel(SigmoidClass(h.a.value, h.c.value), h.f)
end


#===================================================================================================
  RealKernel Product and Sum
===================================================================================================#

# RealKernel Product

immutable KernelProduct{T<:AbstractFloat} <: PointwiseKernel{T}
    a::HyperParameter{T}
    f::RealKernel{T}
    g::RealKernel{T}
    function KernelProduct(a::Variable{T}, f::RealKernel{T}, g::RealKernel)
        new(HyperParameter(a, leftbounded(zero(T), :open)), f, g)
    end
end
function KernelProduct{T<:AbstractFloat}(a::Argument{T}, f::RealKernel{T}, g::RealKernel{T})
    KernelProduct{T}(Variable(a), f, g)
end


# RealKernel Sum

immutable KernelSum{T<:AbstractFloat} <: PointwiseKernel{T}
    c::HyperParameter{T}
    f::RealKernel{T}
    g::RealKernel{T}
    function KernelSum(c::Variable{T}, f::RealKernel{T}, g::RealKernel{T})
        new(HyperParameter(c, leftbounded(zero(T), :closed)), f, g)
    end
end
function KernelSum{T<:AbstractFloat}(c::Argument{T}, f::RealKernel{T}, g::RealKernel{T})
    KernelSum{T}(Variable(c), f, g)
end


# Common Kernels

for (h_obj, h_op, identity, scalar) in (
        (:KernelProduct, :*, :1, :a),
        (:KernelSum,     :+, :0, :c)
    )
    other_identity = identity == :1 ? :0 : :1
    scalar_str = string(scalar)
    @eval begin
        function description_string(h::$h_obj, showtype::Bool = true)
            constant_str = string($scalar_str,"=", h.$scalar.value)
            f_str = "f=" * description_string(h.f, false)
            g_str = "g=" * description_string(h.g, false)
            obj_str = string($h_obj.name.name, showtype ? string("{", eltype(h), "}") : "")
            string(obj_str, "(", constant_str, ",", f_str, ",", g_str, ")")
        end

        function ==(f::$h_obj, g::$h_obj)
            (f.$scalar == g.$scalar) && (f.f == g.f) && (f.g == g.g)
        end

        function convert{T<:AbstractFloat}(::Type{($h_obj){T}}, h::$h_obj)
            $h_obj(Variable(convert(T, h.$scalar.value), h.$scalar.isfixed),
                           convert(RealKernel{T}, h.f), convert(RealKernel{T}, h.g))
        end

        ismercer(h::$h_obj) = ismercer(h.f) && ismercer(h.g)
        isnegdef(h::$h_obj) = isnegdef(h.f) && isnegdef(h.g)
    end
end

for (h_obj, h_op, identity, scalar, op2_identity, op2_scalar) in (
        (:KernelProduct, :*, :1, :a, :0, :c),
        (:KernelSum,     :+, :0, :c, :1, :a)
    )
    @eval begin

        function $h_op{T}($scalar::Real, h::$h_obj{T}) 
            $h_obj($h_op(convert(T, $scalar), h.$scalar.value), h.f, h.g)
        end
        $h_op(h::$h_obj, $scalar::Real) = $h_op($scalar, h)

        $h_op{T}(f::RealKernel{T}, g::RealKernel{T}) = $h_obj(convert(T, $identity), f, g)

        function ($h_op){T}(f::AffineKernel{T}, g::AffineKernel{T})
            if f.$op2_scalar == $op2_identity && g.$op2_scalar == $op2_identity
                $h_obj($h_op(f.$scalar.value, g.$scalar.value), f.f, g.f)
            else
                $h_obj(convert(T, $identity), f, g)
            end
        end

        function ($h_op){T}(f::AffineKernel{T}, g::RealKernel{T})
            if f.$op2_scalar == $op2_identity
                $h_obj(f.$scalar.value, f.f, g)
            else
                $h_obj(convert(T, $identity), f, g)
            end
        end

        function ($h_op){T}(f::RealKernel{T}, g::AffineKernel{T})
            if g.$op2_scalar == $op2_identity
                $h_obj(g.$scalar.value, f, g.f)
            else
                $h_obj(convert(T, $identity), f, g)
            end
        end
    end
end
