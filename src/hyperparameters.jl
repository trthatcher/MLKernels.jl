#=================
  Bounds Objects
=================#

abstract Bounds{T<:Real}

immutable NullBound{T<:Real} <: Bounds{T} end

immutable LowerBound{T<:Real,lstrict} <: Bounds{T}
    lower::T
end
LowerBound{T<:Real}(lstrict::Bool, lower::T) = LowerBound{T,lstrict}(lower)

immutable UpperBound{T<:Real,ustrict} <: Bounds{T}
    upper::T
end
UpperBound{T<:Real}(ustrict::Bool, upper::T) = UpperBound{T,ustrict}(upper)

immutable Interval{T<:Real,lstrict,ustrict} <: Bounds{T}
    lower::T
    upper::T
    function Interval(lower::T, upper::T)
        if (lower > upper) || ((lstrict || ustrict) && lower == upper)
            error("Invalid bounds.")
        end
        new(lower, upper)
    end
end
function Interval{T<:Real}(lstrict::Bool, lower::T, ustrict::Bool, upper::T)
    Interval{T,lstrict,ustrict}(lower, upper)
end

@inline boundstring{T<:AbstractFloat}(B::NullBound{T}) = "ℝ"
@inline boundstring{T<:Integer}(B::NullBound{T}) = "ℤ"

@inline boundstring{T,L}(I::LowerBound{T,L})  = (L ? "(" : "[") * "$(I.lower),∞)"
@inline boundstring{T,U}(I::UpperBound{T,U})  = "(-∞,$(I.upper)" * (U ? ")" : "]")

@inline function boundstring{T,L,U}(I::Interval{T,L,U})
    (L ? "(" : "[") * "$(I.lower),$(I.upper)" * (U ? ")" : "]")
end

function show(io::IO, I::Bounds)
    print(io, "Interval" * boundstring(I))
end

@inline checkbounds{T<:Real}(I::NullBound{T}, x::T) = true

@inline checkbounds{T<:Real}(I::LowerBound{T,true},  x::T) = I.lower <  x
@inline checkbounds{T<:Real}(I::LowerBound{T,false}, x::T) = I.lower <= x

@inline checkbounds{T<:Real}(I::UpperBound{T,true},  x::T) = x <  I.upper
@inline checkbounds{T<:Real}(I::UpperBound{T,false}, x::T) = x <= I.upper

@inline checkbounds{T<:Real}(I::Interval{T,true,true},   x::T) = I.lower <  x <  I.upper
@inline checkbounds{T<:Real}(I::Interval{T,true,false},  x::T) = I.lower <  x <= I.upper
@inline checkbounds{T<:Real}(I::Interval{T,false,true},  x::T) = I.lower <= x <  I.upper
@inline checkbounds{T<:Real}(I::Interval{T,false,false}, x::T) = I.lower <= x <= I.upper

# \BbbR
# \BbbZ

for (sym, data) in ((:ℝ, AbstractFloat), (:ℤ, Integer))
    @eval begin
        function ($sym){T<:$data}(bound::Symbol, value::T)
            if bound == :<
                UpperBound(true, value)
            elseif bound == :(<=)
                UpperBound(false, value)
            elseif bound == :>
                LowerBound(true, value)
            elseif bound == :(>=)
                LowerBound(false, value)
            else
                error("Unrecognized symbol; only :<, :>, :(>=) and :(<=) are accepted.")
            end
        end

        function ($sym){T<:$data}(lbound::Symbol, lower::T, ubound::Symbol, upper::T)
            if lbound == :>
                if ubound == :<
                    Interval(true, lower, true, upper)
                elseif ubound == :(<=)
                    Interval(true, lower, false, upper)
                else
                    error("Unrecognized symbol; only :< and :(<=) are accepted for upper bound.")
                end
            elseif bound == :(>=)
                error("IMPLEMENT ME")
            else
                error("Unrecognized symbol; only :> and :(>=) are accepted for lower bound.")
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

eltype{T<:Real}(::Fixed{T}) = T

typealias Variable{T<:Real} Union{Fixed{T},T}

type Parameter{T<:Real,fixed}
    value::T
    bounds::Bounds{T}
    function Parameter(value::T, bounds::Bounds{T})
        if !checkbounds(bounds, value)
            error("Value $(value) must be in range " * boundstring(bounds))
        end
        new(value, bounds)
    end
end
Parameter{T<:Real}(x::T, bounds::Bounds{T} = NullBound{T}()) = Parameter{T,false}(x, bounds)
function Parameter{T<:Real}(x::Fixed{T}, bounds::Bounds{T} = NullBound{T}())
    Parameter{T,true}(x.value, bounds)
end

function show(io::IO, θ::Parameter)
    print(io, string(θ.value) * " ∈ " * boundstring(θ.bounds))
end

isfixed{T<:Real}(::Parameter{T,true})  = true
isfixed{T<:Real}(::Parameter{T,false}) = false

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
