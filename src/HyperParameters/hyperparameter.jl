#== HyperParameter Type ==#

immutable HyperParameter{T<:Real}
    value::Base.RefValue{T}
    interval::Interval{T}
    function HyperParameter(x::T, I::Interval{T})
        checkvalue(I, x) || error("Value $(x) must be in range " * string(I))
        new(Ref(x), I)
    end
end
HyperParameter{T<:Real}(x::T, I::Interval{T} = interval(T)) = HyperParameter{T}(x, I)

eltype{T}(::HyperParameter{T}) = T

@inline getvalue{T}(θ::HyperParameter{T}) = getindex(θ.value)

function setvalue!{T}(θ::HyperParameter{T}, x::T)
    checkvalue(θ.interval, x) || error("Value $(x) must be in range " * string(θ.interval))
    setindex!(θ.value, x)
    return θ
end

checkvalue{T}(θ::HyperParameter{T}, x::T) = checkvalue(θ.interval, x)

convert{T<:Real}(::Type{HyperParameter{T}}, θ::HyperParameter{T}) = θ
function convert{T<:Real}(::Type{HyperParameter{T}}, θ::HyperParameter)
    HyperParameter{T}(convert(T, getvalue(θ)), convert(Interval{T}, θ.bounds))
end

function show{T}(io::IO, θ::HyperParameter{T})
    print(io, string("HyperParameter(", getvalue(θ), ",", string(θ.interval), ")"))
end

gettheta(θ::HyperParameter) = theta(θ.interval, getvalue(θ))

settheta!{T}(θ::HyperParameter, x::T) = setvalue!(θ, invtheta(θ.interval,x))

checktheta{T}(θ::HyperParameter, x::T) = checkvalue(θ, theta(θ.interval,x))

for op in (:isless, :(==), :+, :-, :*, :/, :^)
    @eval begin
        $op(θ1::HyperParameter, θ2::HyperParameter) = $op(getvalue(θ1), getvalue(θ2))
        $op(a::Number, θ::HyperParameter) = $op(a, getvalue(θ))
        $op(θ::HyperParameter, a::Number) = $op(getvalue(θ), a)
    end
end
