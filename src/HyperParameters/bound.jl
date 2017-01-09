#== Bound Type ==#

immutable Bound{T<:Real}
    value::T
    isopen::Bool
end

function Bound{T<:Real}(value::T, boundtype::Symbol)
    if boundtype == :open
        return Bound(value, true)
    elseif boundtype == :closed
        return Bound(value, false)
    else
        error("Bound type $boundtype not recognized")
    end
end

eltype{T<:Real}(::Bound{T}) = T
convert{T<:Real}(::Type{Bound{T}}, B::Bound) = Bound(convert(T, B.value), B.isopen)
