#=================
  Bounds Objects
=================#

immutable Bound{T<:Real}
    value::T
    is_strict::Bool
end
Bound{T<:Real}(value::T, is_strict::Bool = true) = Bound{T}(value, is_strict)

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

function Interval{T<:Real}(bound::Bound{T}, is_lower::Bool = true)
    if is_lower
        Interval(Nullable(bound), Nullable{Bound{T}}())
    else
        Interval(Nullable{Bound{T}}(), Nullable(bound))
    end
end

unbounded{T<:Real}(::Type{T}) = Interval(Nullable{Bound{T}}(), Nullable{Bound{T}}())

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

immutable Fixed{T<:Real}
    value::T
end
Fixed{T<:Real}(v::T) = Fixed{T}(v)
Fixed{T<:Real}(v::Fixed{T}) = v

eltype{T<:Real}(::Fixed{T}) = T

typealias Variable{T<:Real} Union{Fixed{T},T}

type Parameter{T<:Real}
    value::T
    bounds::Interval{T}
    isfixed::Bool
    function Parameter(value::T, bounds::Interval{T}, isfixed::Bool)
        checkbounds(bounds, value) || error("Value $(value) must be in range " * string(bounds))
        new(value, bounds, isfixed)
    end
end
function Parameter{T<:Real}(x::T, bounds::Interval{T} = unbounded(T), isfixed::Bool=false)
    Parameter{T}(x, bounds, isfixed)
end
function Parameter{T<:Real}(x::Fixed{T}, bounds::Interval{T} = unbounded(T))
    Parameter(x.value, bounds, true)
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
