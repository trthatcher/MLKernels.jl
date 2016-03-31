#=================
  Bounds Objects
=================#

immutable Bound{T<:Real}
    value::T
    is_strict::Bool
end
Bound{T<:Real}(value::T, is_strict::Bool) = Bound{T}(value, is_strict)

function Bound{T<:Real}(value::T, boundtype::Symbol)
    if boundtype == :strict
        return Bound(value, true)
    elseif boundtype == :nonstrict
        return Bound(value, false)
    else
        error("Bound type $boundtype not recognized")
    end
end

eltype{T<:Real}(::Bound{T}) = T

convert{T<:Real}(::Type{Bound{T}}, B::Bound) = Bound(convert(T, B.value), B.is_strict)

immutable Interval{T<:Real}
    lower::Nullable{Bound{T}}
    upper::Nullable{Bound{T}}
    function Interval(lower::Nullable{Bound{T}}, upper::Nullable{Bound{T}})
        if !isnull(lower) && !isnull(upper)
            ubound = get(upper)
            lbound = get(lower)
            if lbound.is_strict || ubound.is_strict
                lbound.value <  ubound.value || error("Invalid bounds")
            else
                lbound.value <= ubound.value || error("Invalid bounds")
            end
        end
        new(lower, upper)
    end
end
Interval{T<:Real}(lower::Nullable{Bound{T}}, upper::Nullable{Bound{T}}) = Interval{T}(lower, upper)
Interval{T<:Real}(lower::Bound{T}, upper::Bound{T}) = Interval(Nullable(lower), Nullable(upper))

eltype{T}(::Interval{T}) = T

function UpperBound{T<:Real}(value::T, boundtype::Symbol)
    if boundtype == :strict
        return Interval(Nullable{Bound{T}}(), Nullable(Bound(value, true)))
    elseif boundtype == :nonstrict
        return Interval(Nullable{Bound{T}}(), Nullable(Bound(value, false)))
    else
        error("Bound type $boundtype not recognized")
    end
end
UpperBound{T<:Real}(upper::Bound{T}) = Interval(Nullable{Bound{T}}(), Nullable(upper))

function LowerBound{T<:Real}(value::T, boundtype::Symbol)
    if boundtype == :strict
        return Interval(Nullable(Bound(value, true)), Nullable{Bound{T}}())
    elseif boundtype == :nonstrict
        return Interval(Nullable(Bound(value, false)), Nullable{Bound{T}}())
    else
        error("Bound type $boundtype not recognized")
    end
end
LowerBound{T<:Real}(lower::Bound{T}) = Interval(Nullable(lower), Nullable{Bound{T}}())

NullBound{T<:Real}(::Type{T}) = Interval(Nullable{Bound{T}}(), Nullable{Bound{T}}())

function convert{T<:Real}(::Type{Interval{T}}, I::Interval)
    if isnull(I.lower)
        isnull(I.upper) ? NullBound(T) : UpperBound(convert(Bound{T}, get(I.upper)))
    else
        if isnull(I.upper)
            LowerBound(convert(Bound{T}, get(I.lower)))
        else
            Interval(convert(Bound{T}, get(I.lower)), convert(Bound{T}, get(I.upper)))
        end
    end
end


function description_string{T}(I::Interval{T})
    interval =  string("Interval{", T, "}")
    if isnull(I.lower)
        if isnull(I.upper)
            string(interval, "(-∞,∞)")
        else
            string(interval, "(-∞,", get(I.upper).value, get(I.upper).is_strict ? ")" : "]")
        end
    else
        lower = string(get(I.lower).is_strict ? "(" : "[",  get(I.lower).value, ",")
        if isnull(I.upper)
            string(interval, lower, "∞)")
        else
            string(interval, lower, get(I.upper).value, get(I.upper).is_strict ? ")" : "]")
        end
    end
end
function show{T}(io::IO, I::Interval{T})
    print(io, description_string(I))
end


function checkbounds{T<:Real}(I::Interval{T}, x::T)
    if isnull(I.lower)
        if isnull(I.upper)
            true
        else
            get(I.upper).is_strict ? (x < get(I.upper).value) : (x <= get(I.upper).value)
        end
    else
        lb = get(I.lower)
        if isnull(I.upper)
            lb.is_strict ? (lb.value < x) : (lb.value <= x)
        else
            if (ub = get(I.upper)).is_strict
                lb.is_strict ? (lb.value < x < ub.value) : (lb.value <= x < ub.value)
            else
                lb.is_strict ? (lb.value < x <= ub.value) : (lb.value <= x <= ub.value)
            end
        end
    end
end


#========================
  HyperParameter Object
========================#

immutable Variable{T<:Real}
    value::T
    isfixed::Bool
end
Variable{T<:AbstractFloat}(value::T, isfixed::Bool=false) = Variable{T}(value, false)
Fixed{T<:Real}(v::T) = Variable{T}(v, true)

eltype{T<:Real}(::Variable{T}) = T

function convert{T<:Real}(::Type{Variable{T}}, var::Variable)
    Variable(convert(T, var.value), var.isfixed)
end
convert{T<:Real}(::Type{Variable{T}}, value::Real) = Variable(convert(T, value), false)

typealias Argument{T<:Real} Union{T,Variable{T}}


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

function show{T}(io::IO, θ::HyperParameter{T})
    print(io, "HyperParameter{" * string(T) * "}(", θ.value, ") ∈ ", θ.bounds)
end

isfixed(θ::HyperParameter) = θ.isfixed

function Variable{T<:Real}(θ::HyperParameter{T})
    Variable(θ.value, θ.isfixed)
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
