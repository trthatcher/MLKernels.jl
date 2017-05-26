immutable Interval{T<:Real,A<:Bound,B<:Bound}
    a::A
    b::B
    function Interval(a::Bound{T}, b::Bound{T})
        if !(A <: NullBound || B <: NullBound)
            va = a.value
            vb = b.value
            if A <: ClosedBound && B <: ClosedBound
                va <= vb || error("Invalid bounds: a=$va must be less than or equal to b=$vb")
            else
                va < vb || error("Invalid bounds: a=$va must be less than b=$vb")
            end
        end
        new(a,b)
    end
end
Interval{T<:Real}(a::Bound{T}, b::Bound{T}) = Interval{T,typeof(a),typeof(b)}(a,b)

eltype{T}(::Interval{T}) = T

interval(a::Void, b::Void) = Interval(NullBound{Float64}(), NullBound{Float64}())
interval{T<:Real}(a::Bound{T}, b::Void) = Interval(a, NullBound{T}())
interval{T<:Real}(a::Void, b::Bound{T}) = Interval(NullBound{T}(), b)
interval{T<:Real}(::Type{T}) = Interval(NullBound{T}(), NullBound{T}())
interval{T<:Real}(a::Bound{T}, b::Bound{T}) = Interval(a,b)

checkvalue(I::Interval, x::Real) = checkvalue(I.a, x) && checkvalue(x, I.b)

function theta{T<:AbstractFloat}(I::Interval{T,OpenBound{T},OpenBound{T}}, x::T)
    y = (x - I.a.value)/(I.b.value - I.a.value)
    log(y/(one(T)-y))
end
theta{T<:AbstractFloat,_<:Bound}(I::Interval{T,OpenBound{T},_}, x::T) = log(x - I.a.value)
theta{T<:AbstractFloat,_<:Bound}(I::Interval{T,_,OpenBound{T}}, x::T) = log(I.b.value - x)
theta{T<:AbstractFloat}(I::Interval{T}, x::T) = x

function invtheta{T<:AbstractFloat}(I::Interval{T,OpenBound{T},OpenBound{T}}, x::T)
    y = exp(x)/(one(T)+exp(x))
    (I.b.value - I.a.value)*y + I.a.value
end
invtheta{T<:AbstractFloat,_<:Bound}(I::Interval{T,OpenBound{T},_}, x::T) = exp(x) + I.a.value
invtheta{T<:AbstractFloat,_<:Bound}(I::Interval{T,_,OpenBound{T}}, x::T) = I.b.value - exp(x)
invtheta{T<:AbstractFloat}(I::Interval{T}, x::T) = x

#=
upperbound{T}(I::Interval{T,OpenBound{T},OpenBound{T}}) = convert(T, Inf)
upperbound{T<:AbstractFloat,_<:Bound}(I::Interval{T,OpenBound{T},_}) = log(x - I.a.value)
upperbound{T<:AbstractFloat,_<:Bound}(I::Interval{T,_,OpenBound{T}}) = log(I.b.value - x)
upperbound{T<:AbstractFloat}(I::Interval{T}, x::T) = x
=#

function string{T1,T2,T3}(I::Interval{T1,T2,T3})
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
