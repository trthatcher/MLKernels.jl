module HyperParameters

import Base: convert, eltype, show, ==, *, /, +, -, ^, besselk, exp, gamma, tanh

export Bound, Interval, leftbounded, rightbounded, unbounded, checkbounds, Variable, fixed, 
    Argument, HyperParameter


#== Bound Type ==#

immutable Bound{T<:Real}
    value::T
    isopen::Bool
end
Bound{T<:Real}(value::T, isopen::Bool) = Bound{T}(value, isopen)

function Bound{T<:Real}(value::T, boundtype::Symbol)
    if boundtype == :open
        return Bound(value, true)
    elseif boundtype == :closed
        return Bound(value, false)
    else
        error("Bound type $boundtype not recognized")
    end
end

eltype{T<:Real}(::Bound{T}) = T
convert{T<:Real}(::Type{Bound{T}}, B::Bound) = Bound(convert(T, B.value), B.isopen)


#== Interval Type ==#

immutable Interval{T<:Real}
    left::Nullable{Bound{T}}
    right::Nullable{Bound{T}}
    function Interval(left::Nullable{Bound{T}}, right::Nullable{Bound{T}})
        if !isnull(left) && !isnull(right)
            rbound = get(right)
            lbound = get(left)
            if lbound.isopen || rbound.isopen
                lbound.value <  rbound.value || error("Invalid bounds")
            else
                lbound.value <= rbound.value || error("Invalid bounds")
            end
        end
        new(left, right)
    end
end
Interval{T<:Real}(left::Nullable{Bound{T}}, right::Nullable{Bound{T}}) = Interval{T}(left, right)
Interval{T<:Real}(left::Bound{T}, right::Bound{T}) = Interval(Nullable(left), Nullable(right))

eltype{T}(::Interval{T}) = T

function leftbounded{T<:Real}(value::T, boundtype::Symbol)
    if boundtype == :open
        Interval(Nullable(Bound(value, true)),  Nullable{Bound{T}}())
    elseif boundtype == :closed
        Interval(Nullable(Bound(value, false)), Nullable{Bound{T}}())
    else
        error("Bound type $boundtype not recognized")
    end
end
leftbounded{T<:Real}(left::Bound{T}) = Interval(Nullable(left), Nullable{Bound{T}}())

function rightbounded{T<:Real}(value::T, boundtype::Symbol)
    if boundtype == :open
        Interval(Nullable{Bound{T}}(), Nullable(Bound(value, true)))
    elseif boundtype == :closed
        Interval(Nullable{Bound{T}}(), Nullable(Bound(value, false)))
    else
        error("Bound type $boundtype not recognized")
    end
end
rightbounded{T<:Real}(right::Bound{T}) = Interval(Nullable{Bound{T}}(), Nullable(right))


unbounded{T<:Real}(::Type{T}) = Interval(Nullable{Bound{T}}(), Nullable{Bound{T}}())

function convert{T<:Real}(::Type{Interval{T}}, I::Interval)
    if isnull(I.left)
        isnull(I.right) ? unbounded(T) : rightbounded(convert(Bound{T}, get(I.right)))
    else
        if isnull(I.right)
            leftbounded(convert(Bound{T}, get(I.left)))
        else
            Interval(convert(Bound{T}, get(I.left)), convert(Bound{T}, get(I.right)))
        end
    end
end


function description_string{T}(I::Interval{T})
    interval =  string("Interval{", T, "}")
    if isnull(I.left)
        if isnull(I.right)
            string(interval, "(-∞,∞)")
        else
            string(interval, "(-∞,", get(I.right).value, get(I.right).isopen ? ")" : "]")
        end
    else
        left = string(get(I.left).isopen ? "(" : "[",  get(I.left).value, ",")
        if isnull(I.right)
            string(interval, left, "∞)")
        else
            string(interval, left, get(I.right).value, get(I.right).isopen ? ")" : "]")
        end
    end
end
function show{T}(io::IO, I::Interval{T})
    print(io, description_string(I))
end


function checkbounds{T<:Real}(I::Interval{T}, x::T)
    if isnull(I.left)
        if isnull(I.right)
            true
        else
            ub = get(I.right)
            ub.isopen ? (x < ub.value) : (x <= ub.value)
        end
    else
        lb = get(I.left)
        if isnull(I.right)
            lb.isopen ? (lb.value < x) : (lb.value <= x)
        else
            ub = get(I.right)
            if ub.isopen
                lb.isopen ? (lb.value < x < ub.value) : (lb.value <= x < ub.value)
            else
                lb.isopen ? (lb.value < x <= ub.value) : (lb.value <= x <= ub.value)
            end
        end
    end
end


#== Variable Type ==#

immutable Variable{T<:Real}
    value::T
    isfixed::Bool
end
Variable{T<:Real}(value::T, isfixed::Bool=false) = Variable{T}(value, false)
Variable{T<:Real}(value::Variable{T}) = value
fixed{T<:Real}(v::T) = Variable{T}(v, true)

eltype{T<:Real}(::Variable{T}) = T

convert{T<:Real}(::Type{Variable{T}}, var::Variable) = Variable(convert(T, var.value), var.isfixed)
convert{T<:Real}(::Type{Variable{T}}, var::Real) = Variable(convert(T, var), false)

typealias Argument{T<:Real} Union{T,Variable{T}}


#== HyperParameter Type ==#

type HyperParameter{T<:Real}
    value::T
    bounds::Interval{T}
    isfixed::Bool
    function HyperParameter(value::T, bounds::Interval{T}, isfixed::Bool)
        checkbounds(bounds, value) || error("Value $(value) must be in range " * string(bounds))
        new(value, bounds, isfixed)
    end
end
function HyperParameter{T<:Real}(x::T, bounds::Interval{T} = NullBound(T), isfixed::Bool=false)
    HyperParameter{T}(x, bounds, isfixed)
end
function HyperParameter{T<:Real}(x::Variable{T}, bounds::Interval{T} = NullBound(T))
    HyperParameter(x.value, bounds, x.isfixed)
end

function convert{T<:Real}(::Type{HyperParameter{T}}, θ::HyperParameter)
    HyperParameter{T}(convert(T, θ.value), convert(Interval{T}, θ.bounds), θ.isfixed)
end

function convert{T<:Real}(::Type{Variable{T}}, θ::HyperParameter)
    Variable{T}(convert(T, θ.value), θ.isfixed)
end

function show{T}(io::IO, θ::HyperParameter{T})
    print(io, "HyperParameter{" * string(T) * "}(", θ.value, ") ∈ ", θ.bounds)
end

@inline *(a::Real, v::HyperParameter) = *(a, v.value)
@inline *(v::HyperParameter, a::Real) = *(v.value, a)

@inline /(a::Real, v::HyperParameter) = /(a, v.value)
@inline /(v::HyperParameter, a::Real) = /(v.value, a)

@inline +(a::Real, v::HyperParameter) = +(a, v.value)
@inline +(v::HyperParameter, a::Real) = +(v.value, a)

@inline -(v::HyperParameter) = -(v.value)
@inline -(a::Real, v::HyperParameter) = -(a, v.value)
@inline -(v::HyperParameter, a::Real) = -(v.value, a)

@inline ^(a::Real, v::HyperParameter)          = ^(a, v.value)
@inline ^(v::HyperParameter, a::Integer)       = ^(v.value, a)
@inline ^(v::HyperParameter, a::AbstractFloat) = ^(v.value, a)

@inline besselk(v::HyperParameter, x::Real) = besselk(v.value, x)
@inline exp(v::HyperParameter)  = exp(v.value)
@inline gamma(v::HyperParameter) = gamma(v.value)
@inline tanh(v::HyperParameter) = tanh(v.value)

@inline ==(a::Real, v::HyperParameter) = ==(a, v.value)
@inline ==(v::HyperParameter, a::Real) = ==(v.value, a)
@inline ==(v1::HyperParameter, v2::HyperParameter) = ==(v1.value, v2.value)

end # End HyperParameter
