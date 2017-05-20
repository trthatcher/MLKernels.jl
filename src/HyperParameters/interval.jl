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

checkvalue(I::Interval, x::Real) = checkvalue(I.a, x) && checkvalue(x, I.b)

function link{T<:AbstractFloat}(I::Interval{T,OpenBound{T},OpenBound{T}}, x::T)
    y = (z - I.a.value)/(I.b.value - I.a.value)
    log(y/(1-y))
end
link{T<:AbstractFloat}(I::Interval{T,OpenBound{T},Bound{T}}, x::T) = log(x - I.a.value)
link{T<:AbstractFloat}(I::Interval{T,Bound{T},OpenBound{T}}, x::T) = log(I.b.value - x)
link{T<:AbstractFloat}(I::Interval{T}, x::T) = x

interval(a::Void, b::Void) = Interval(NullBound{Float64}(), NullBound{Float64}())
interval{T<:Real}(a::Bound{T}, b::Void) = Interval(a, NullBound{T}())
interval{T<:Real}(a::Void, b::Bound{T}) = Interval(NullBound{T}(), b)
interval{T<:Real}(::Type{T}) = Interval(NullBound{T}(), NullBound{T}())
interval{T<:Real}(a::Bound{T}, b::Bound{T}) = Interval(a,b)

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
