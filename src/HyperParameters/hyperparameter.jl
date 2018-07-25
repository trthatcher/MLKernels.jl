#== HyperParameter Type ==#

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

gettheta(θ::HyperParameter) = theta(θ.interval, getvalue(θ))

settheta!(θ::HyperParameter, x::T) where {T} = setvalue!(θ, eta(θ.interval,x))

checktheta(θ::HyperParameter, x::T) where {T} = checktheta(θ.interval, x)

for op in (:isless, :(==), :+, :-, :*, :/)
    @eval begin
        $op(θ1::HyperParameter, θ2::HyperParameter) = $op(getvalue(θ1), getvalue(θ2))
        $op(a::Number, θ::HyperParameter) = $op(a, getvalue(θ))
        $op(θ::HyperParameter, a::Number) = $op(getvalue(θ), a)
    end
end
