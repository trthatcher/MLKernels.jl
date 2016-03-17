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

convert{T<:Real}(::Type{Bound{T}}, B::Bound) = Bound(convert(T, B.value), B.is_strict)

immutable Interval{T<:Real}
    lower::Nullable{Bound{T}}
    upper::Nullable{Bound{T}}
    function Interval(lower::Nullable{Bound{T}}, upper::Nullable{Bound{T}})
        if !isnull(lower) && !isnull(upper)
            ubound = get(upper)
            if !((lbound = get(lower)).is_strict) && !ubound.is_strict
                lbound.value <= ubound.value || error("Invalid bounds")
            else
                lbound.value <  ubound.value || error("Invalid bounds")
            end
        end
        new(lower, upper)
    end
end
Interval{T<:Real}(lower::Nullable{Bound{T}}, upper::Nullable{Bound{T}}) = Interval{T}(lower, upper)
Interval{T<:Real}(lower::Bound{T}, upper::Bound{T}) = Interval(Nullable(lower), Nullable(upper))

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

   
LowerBound{T<:Real}(x::T) = Interval(Nullable(B), Nullable{Bound{T}}())
NullBound{T<:Real}(::Type{T})    = Interval(Nullable{Bound{T}}(), Nullable{Bound{T}}())

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


function show{T}(io::IO, I::Interval{T})
    if isnull(I.lower)
        if isnull(I.upper)
            print(io, T <: Integer ? "ℤ" : "ℝ")
        else
            print(io, "(-∞,", get(I.upper).value, get(I.upper).is_strict ? ")" : "]")
        end
    else
        if isnull(I.upper)
            print(io, get(I.lower).is_strict ? "(" : "[", get(I.lower).value, ",∞)")
        else
            print(io, get(I.lower).is_strict ? "(" : "[", get(I.lower).value, ",",
                      get(I.upper).value, get(I.upper).is_strict ? ")" : "]")
        end
    end
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
  Hyperparameter Object
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


type Parameter{T<:Real}
    value::T
    bounds::Interval{T}
    isfixed::Bool
    function Parameter(value::T, bounds::Interval{T}, isfixed::Bool)
        checkbounds(bounds, value) || error("Value $(value) must be in range " * string(bounds))
        new(value, bounds, isfixed)
    end
end
function Parameter{T<:Real}(x::T, bounds::Interval{T} = NullBound(T), isfixed::Bool=false)
    Parameter{T}(x, bounds, isfixed)
end
function Parameter{T<:Real}(x::Variable{T}, bounds::Interval{T} = NullBound(T))
    Parameter(x.value, bounds, x.isfixed)
end

function convert{T<:Real}(::Type{Parameter{T}}, θ::Parameter)
    Parameter{T}(convert(T, θ.value), convert(Interval{T}, θ.bounds), θ.isfixed)
end

function show(io::IO, θ::Parameter)
    print(io, "Parameter(", θ.value, ") ∈ ", θ.bounds)
end

isfixed(θ::Parameter) = θ.isfixed

@inline *(a::Real, v::Parameter) = *(a, v.value)
@inline *(v::Parameter, a::Real) = *(v.value, a)

@inline /(a::Real, v::Parameter) = /(a, v.value)
@inline /(v::Parameter, a::Real) = /(v.value, a)

@inline +(a::Real, v::Parameter) = +(a, v.value)
@inline +(v::Parameter, a::Real) = +(v.value, a)

@inline -(v::Parameter) = -(v.value)
@inline -(a::Real, v::Parameter) = -(a, v.value)
@inline -(v::Parameter, a::Real) = -(v.value, a)

@inline ^(a::Real, v::Parameter)          = ^(a, v.value)
@inline ^(v::Parameter, a::Integer)       = ^(v.value, a)
@inline ^(v::Parameter, a::AbstractFloat) = ^(v.value, a)

@inline exp(v::Parameter)  = exp(v.value)
@inline tanh(v::Parameter) = tanh(v.value)

@inline ==(a::Real, v::Parameter) = ==(a, v.value)
@inline ==(v::Parameter, a::Real) = ==(v.value, a)
@inline ==(v1::Parameter, v2::Parameter) = ==(v1.value, v2.value)
