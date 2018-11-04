module HyperParameters

import Base: convert, eltype, promote_type, show, string, ==, *, /, +, -, ^, isless, depwarn

export
    Bound,
        LeftBound,
        RightBound,
        NullBound,

    Interval,
    interval,
    checkbounds,

    HyperParameter,
    getvalue,
    setvalue!,
    checkvalue,
    gettheta,
    settheta!,
    checktheta,
    upperboundtheta,
    lowerboundtheta


#= bound.jl =#
abstract type Bound{T} end

eltype(::Bound{T}) where {T} = T

struct OpenBound{T<:Real} <: Bound{T}
    value::T
    function OpenBound{T}(x::Real) where {T<:Real}
        !(T <: Integer) || error("Bounds must be closed for integers")
        if T <: AbstractFloat
            !isnan(x) || error("Bound value must not be NaN")
            !isinf(x) || error("Bound value must not be Inf/-Inf")
        end
        new{T}(x)
    end
end
OpenBound(x::T) where {T<:Real} = OpenBound{T}(x)

convert(::Type{OpenBound{T}}, b::OpenBound{T}) where {T<:Real} = b
convert(::Type{OpenBound{T}}, b::OpenBound) where {T<:Real} = OpenBound{T}(b.value)
string(b::OpenBound) = string("OpenBound(", b.value, ")")

struct ClosedBound{T<:Real} <: Bound{T}
    value::T
    function ClosedBound{T}(x::Real) where {T<:Real}
        if T <: AbstractFloat
            !isnan(x) || error("Bound value must not be NaN")
            !isinf(x) || error("Bound value must not be Inf/-Inf")
        end
        new{T}(x)
    end
end
ClosedBound(x::T) where {T<:Real} = ClosedBound{T}(x)

convert(::Type{ClosedBound{T}}, b::ClosedBound{T}) where {T<:Real} = b
convert(::Type{ClosedBound{T}}, b::ClosedBound) where {T<:Real} = ClosedBound{T}(b.value)
string(b::ClosedBound) = string("ClosedBound(", b.value, ")")


struct NullBound{T<:Real} <: Bound{T} end
NullBound(::Type{T}) where {T<:Real} = NullBound{T}()

convert(::Type{NullBound{T}}, b::NullBound{T}) where {T<:Real} = b
convert(::Type{NullBound{T}}, b::NullBound) where {T<:Real} = NullBound{T}()
string(b::NullBound{T}) where {T} = string("NullBound(", T, ")")


checkvalue(a::NullBound,   x::Real) = true
checkvalue(a::OpenBound,   x::Real) = a.value <  x
checkvalue(a::ClosedBound, x::Real) = a.value <= x

checkvalue(x::Real, a::NullBound)   = true
checkvalue(x::Real, a::OpenBound)   = x <  a.value
checkvalue(x::Real, a::ClosedBound) = x <= a.value

promote_rule(::Type{Bound{T1}},::Type{Bound{T2}}) where {T1,T2} = Bound{promote_rule(T1,T2)}

function show(io::IO, b::T) where {T<:Bound}
    print(io, string(b))
end

#= interval.jl =#

struct Interval{T<:Real,A<:Bound{T},B<:Bound{T}}
    a::A
    b::B
    function Interval{T}(a::A, b::B) where {T<:Real,A<:Bound{T},B<:Bound{T}}
        if !(A <: NullBound || B <: NullBound)
            va = a.value
            vb = b.value
            if A <: ClosedBound && B <: ClosedBound
                va <= vb || error("Invalid bounds: a=$va must be less than or equal to b=$vb")
            else
                va < vb || error("Invalid bounds: a=$va must be less than b=$vb")
            end
        end
        new{T,A,B}(a,b)
    end
end
Interval(a::Bound{T}, b::Bound{T}) where {T<:Real} = Interval{T}(a,b)

eltype(::Interval{T}) where {T} = T

interval(a::Nothing, b::Nothing) = Interval(NullBound{Float64}(), NullBound{Float64}())
interval(a::Bound{T}, b::Nothing) where {T<:Real} = Interval(a, NullBound{T}())
interval(a::Nothing, b::Bound{T}) where {T<:Real} = Interval(NullBound{T}(), b)
interval(a::Bound{T}, b::Bound{T}) where {T<:Real} = Interval(a,b)
interval(::Type{T}) where {T<:Real} = Interval(NullBound{T}(), NullBound{T}())


checkvalue(I::Interval, x::Real) = checkvalue(I.a, x) && checkvalue(x, I.b)

function theta(I::Interval{T,A,B}, x::T) where {T<:AbstractFloat,A,B}
    depwarn("theta will be removed entirely in a future release", :theta)
    checkvalue(I,x) || throw(DomainError(x, "Not in $I"))
    if A <: OpenBound
        return B <: OpenBound ? log(x-I.a.value) - log(I.b.value-x) : log(x-I.a.value)
    else
        return B <: OpenBound ? log(I.b.value-x) : x
    end
end

function upperboundtheta(I::Interval{T,A,B}) where {T<:AbstractFloat,A,B}
    if B <: ClosedBound
        return A <: OpenBound ? log(I.b.value - I.a.value) : I.b.value
    elseif B <: OpenBound
        return A <: ClosedBound ? log(I.b.value - I.a.value) : convert(T,Inf)
    else
        return convert(T,Inf)
    end
end

function lowerboundtheta(I::Interval{T,A,B}) where {T<:AbstractFloat,A,B}
    A <: ClosedBound && !(B <: OpenBound) ? I.a.value : convert(T,-Inf)
end

function checktheta(I::Interval{T}, x::T) where {T<:AbstractFloat}
    lowerboundtheta(I) <= x <= upperboundtheta(I)
end

function eta(I::Interval{T,A,B}, x::T) where {T<:AbstractFloat,A,B}
    depwarn("eta will be removed entirely in a future release", :eta)
    checktheta(I,x) || throw(DomainError(x, "Not in $I"))
    if A <: OpenBound
        if B <: OpenBound
            return (I.b.value*exp(x) + I.a.value)/(one(T) + exp(x))
        else
            return exp(x) + I.a.value
        end
    else
        return B <: OpenBound ? I.b.value - exp(x) : x
    end
end

function string(I::Interval{T1,T2,T3}) where {T1,T2,T3}
    if T2 <: NullBound
        if T3 <: NullBound
            string("interval(", T1, ")")
        else
            string("interval(nothing,", string(I.b), ")")
        end
    else
        string("interval(", string(I.a), ",", T3 <: NullBound ? "nothing" : string(I.b), ")")
    end
end

function show(io::IO, I::Interval)
    print(io, string(I))
end

#= hyperparameter.jl =#

struct HyperParameter{T<:Real}
    value::Base.RefValue{T}
    interval::Interval{T}
    function HyperParameter{T}(x::T, I::Interval{T}) where {T<:Real}
        checkvalue(I, x) || error("Value $(x) must be in range " * string(I))
        new{T}(Ref(x), I)
    end
end
HyperParameter(x::T, I::Interval{T} = interval(T)) where {T<:Real} = HyperParameter{T}(x, I)

eltype(::HyperParameter{T}) where {T} = T

@inline getvalue(θ::HyperParameter{T}) where {T} = getindex(θ.value)

function setvalue!(θ::HyperParameter{T}, x::T) where {T}
    checkvalue(θ.interval, x) || error("Value $(x) must be in range " * string(θ.interval))
    setindex!(θ.value, x)
    return θ
end

checkvalue(θ::HyperParameter{T}, x::T) where {T} = checkvalue(θ.interval, x)

convert(::Type{HyperParameter{T}}, θ::HyperParameter{T}) where {T<:Real} = θ
function convert(::Type{HyperParameter{T}}, θ::HyperParameter) where {T<:Real}
    HyperParameter{T}(convert(T, getvalue(θ)), convert(Interval{T}, θ.bounds))
end

function show(io::IO, θ::HyperParameter{T}) where {T}
    print(io, string("HyperParameter(", getvalue(θ), ",", string(θ.interval), ")"))
end

function gettheta(θ::HyperParameter)
    depwarn("gettheta will be removed entirely in a future release", :gettheta)
    theta(θ.interval, getvalue(θ))
end

function settheta!(θ::HyperParameter, x::T) where {T}
    depwarn("settheta! will be removed entirely in a future release", :(settheta!))
    setvalue!(θ, eta(θ.interval,x))
end

function checktheta(θ::HyperParameter, x::T) where {T}
    depwarn("checktheta will be removed entirely in a future release", :checktheta)
    checktheta(θ.interval, x)
end

for op in (:isless, :(==), :+, :-, :*, :/)
    @eval begin
        $op(θ1::HyperParameter, θ2::HyperParameter) = $op(getvalue(θ1), getvalue(θ2))
        $op(a::Number, θ::HyperParameter) = $op(a, getvalue(θ))
        $op(θ::HyperParameter, a::Number) = $op(getvalue(θ), a)
    end
end

end # End HyperParameter