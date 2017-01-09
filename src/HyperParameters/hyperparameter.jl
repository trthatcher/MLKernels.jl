#== HyperParameter Type ==#

type HyperParameter{T<:Real}
    value::T
    bounds::Interval{T}
    function HyperParameter(value::T, bounds::Interval{T})
        checkbounds(bounds, value) || error("Value $(value) must be in range " * string(bounds))
        new(value, bounds)
    end
end
function HyperParameter{T<:Real}(x::T, bounds::Interval{T} = NullBound(T))
    HyperParameter{T}(x, bounds)
end

function convert{T<:Real}(::Type{HyperParameter{T}}, θ::HyperParameter)
    HyperParameter{T}(convert(T, θ.value), convert(Interval{T}, θ.bounds))
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
@inline exp(v::HyperParameter)   = exp(v.value)
@inline gamma(v::HyperParameter) = gamma(v.value)
@inline tanh(v::HyperParameter)  = tanh(v.value)

@inline ==(a::Real, v::HyperParameter) = ==(a, v.value)
@inline ==(v::HyperParameter, a::Real) = ==(v.value, a)
@inline ==(v1::HyperParameter, v2::HyperParameter) = ==(v1.value, v2.value)
