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
