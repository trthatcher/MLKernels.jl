#===================================================================================================
  RealFunction Affinity
===================================================================================================#

abstract PointwiseFunction{T} <: RealFunction{T}

doc"AffineFunction(f;a,c) = a⋅f + c"
immutable AffineFunction{T<:AbstractFloat} <: PointwiseFunction{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    f::RealFunction{T}
    AffineFunction(a::Variable{T}, c::Variable{T}, f::RealFunction{T}) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed)),
        f
    )
end
function AffineFunction{T<:AbstractFloat}(a::Argument{T}, c::Argument{T}, f::RealFunction{T})
    AffineFunction{T}(Variable(a), Variable(c), f)
end

function ==(f::AffineFunction, g::AffineFunction)
    (f.a == g.a) && (f.c == g.c) && (f.f == g.f)
end

ismercer(h::AffineFunction) = ismercer(h.f)
isnegdef(h::AffineFunction) = isnegdef(h.f)

attainszero(h::AffineFunction)     = attainszero(h.f)
attainspositive(h::AffineFunction) = attainspositive(h.f)
attainsnegative(h::AffineFunction) = attainsnegative(h.f)

function description_string(h::AffineFunction, showtype::Bool = true)
    obj_str = string("AffineFunction", showtype ? string("{", eltype(h), "}") : "")
    h_str = description_string(h.f)
    string(obj_str, "(a=$(h.a.value),c=$(h.c.value),", h_str, ")")
end

function convert{T<:AbstractFloat}(::Type{AffineFunction{T}}, h::AffineFunction)
    AffineFunction(convert(T, h.a.value), convert(T, h.c.value), convert(RealFunction{T}, h.f))
end


# Operations

+{T<:AbstractFloat}(f::RealFunction{T}, c::Real) = AffineFunction(one(T), convert(T, c), f)
+(c::Real, f::RealFunction) = +(f, c)

*{T<:AbstractFloat}(f::RealFunction{T}, a::Real) = AffineFunction(convert(T, a), zero(T), f)
*(a::Real, f::RealFunction) = *(f, a)

function +{T<:AbstractFloat}(f::AffineFunction{T}, c::Real)
    AffineFunction(f.a.value, f.c + convert(T,c), f.f)
end
+(c::Real, f::AffineFunction) = +(f, c)

function *{T<:AbstractFloat}(f::AffineFunction{T}, a::Real)
    a = convert(T, a)
    AffineFunction(a * f.a, a * f.c, f.f)
end
*(a::Real, f::AffineFunction) = *(f, a)

function ^{T<:AbstractFloat}(h::AffineFunction{T}, d::Integer)
    CompositeRealFunction(PolynomialClass(h.a.value, h.c.value, convert(T,d)), h.f)
end

function ^{T<:AbstractFloat}(h::AffineFunction{T}, γ::AbstractFloat)
    CompositeRealFunction(PowerClass(h.a.value, h.c.value, convert(T,γ)), h.f)
end

function exp{T<:AbstractFloat}(h::AffineFunction{T})
    CompositeRealFunction(ExponentiatedClass(h.a.value, h.c.value), h.f)
end

function tanh{T<:AbstractFloat}(h::AffineFunction{T})
    CompositeRealFunction(SigmoidClass(h.a.value, h.c.value), h.f)
end


#===================================================================================================
  RealFunction Product and Sum
===================================================================================================#

# RealFunction Product

immutable FunctionProduct{T<:AbstractFloat} <: PointwiseFunction{T}
    a::HyperParameter{T}
    f::RealFunction{T}
    g::RealFunction{T}
    function FunctionProduct(a::Variable{T}, f::RealFunction{T}, g::RealFunction)
        new(HyperParameter(a, leftbounded(zero(T), :open)), f, g)
    end
end
function FunctionProduct{T<:AbstractFloat}(a::Argument{T}, f::RealFunction{T}, g::RealFunction{T})
    FunctionProduct{T}(Variable(a), f, g)
end


# RealFunction Sum

immutable FunctionSum{T<:AbstractFloat} <: PointwiseFunction{T}
    c::HyperParameter{T}
    f::RealFunction{T}
    g::RealFunction{T}
    function FunctionSum(c::Variable{T}, f::RealFunction{T}, g::RealFunction{T})
        new(HyperParameter(c, leftbounded(zero(T), :closed)), f, g)
    end
end
function FunctionSum{T<:AbstractFloat}(c::Argument{T}, f::RealFunction{T}, g::RealFunction{T})
    FunctionSum{T}(Variable(c), f, g)
end


# Common Functions

for (h_obj, h_op, identity, scalar) in (
        (:FunctionProduct, :*, :1, :a),
        (:FunctionSum,     :+, :0, :c)
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
                           convert(RealFunction{T}, h.f), convert(RealFunction{T}, h.g))
        end

        ismercer(h::$h_obj) = ismercer(h.f) && ismercer(h.g)
        isnegdef(h::$h_obj) = isnegdef(h.f) && isnegdef(h.g)
    end
end

for (h_obj, h_op, identity, scalar, op2_identity, op2_scalar) in (
        (:FunctionProduct, :*, :1, :a, :0, :c),
        (:FunctionSum,     :+, :0, :c, :1, :a)
    )
    @eval begin

        function $h_op{T}($scalar::Real, h::$h_obj{T}) 
            $h_obj($h_op(convert(T, $scalar), h.$scalar.value), h.f, h.g)
        end
        $h_op(h::$h_obj, $scalar::Real) = $h_op($scalar, h)

        $h_op{T}(f::RealFunction{T}, g::RealFunction{T}) = $h_obj(convert(T, $identity), f, g)

        function ($h_op){T}(f::AffineFunction{T}, g::AffineFunction{T})
            if f.$op2_scalar == $op2_identity && g.$op2_scalar == $op2_identity
                $h_obj($h_op(f.$scalar.value, g.$scalar.value), f.f, g.f)
            else
                $h_obj(convert(T, $identity), f, g)
            end
        end

        function ($h_op){T}(f::AffineFunction{T}, g::RealFunction{T})
            if f.$op2_scalar == $op2_identity
                $h_obj(f.$scalar.value, f.f, g)
            else
                $h_obj(convert(T, $identity), f, g)
            end
        end

        function ($h_op){T}(f::RealFunction{T}, g::AffineFunction{T})
            if g.$op2_scalar == $op2_identity
                $h_obj(g.$scalar.value, f, g.f)
            else
                $h_obj(convert(T, $identity), f, g)
            end
        end
    end
end
