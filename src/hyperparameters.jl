abstract Bounds{T<:Real}

immutable NullBound{T<:Real} <: Bounds{T} end

immutable LowerBound{T<:Real} <: Bounds{T}
    lstrict::Bool
    lower::T
end
LowerBound{T<:Real}(lstrict::Bool, lower::T) = LowerBound{T}(lstrict, lower)

immutable UpperBound{T<:Real} <: Bounds{T}
    ustrict::Bool
    upper::T
end
UpperBound{T<:Real}(ustrict::Bool, upper::T) = UpperBound{T}(ustrict, upper)

immutable Interval{T<:Real} <: Bounds{T}
    lstrict::Bool
    lower::T
    ustrict::Bool
    upper::T
end
function Interval{T<:Real}(lstrict::Bool, lower::T, ustrict::Bool, upper::T)
    Interval{T}(lstrict, lower, ustrict, upper)
end

@inline boundstring(B::NullBound)  = "(-∞,∞)"
@inline boundstring(I::LowerBound) = (I.lstrict ? "(" : "[") * string(I.lower) * ",∞)"
@inline boundstring(I::UpperBound) = "(-∞," * string(I.upper) * (I.ustrict ? ")" : "]")
@inline function boundstring(I::Interval)
    (I.lstrict ? "(" : "[") * string(I.lower) * "," * string(I.upper) * (I.ustrict ? ")" : "]")
end

function show(io::IO, I::Bounds)
    print(io, "Interval" * boundstring(I))
end

@inline checkbounds{T<:AbstractFloat}(I::NullBound{T}, x::T) = true
@inline checkbounds{T<:AbstractFloat}(I::LowerBound{T}, x::T) = I.lstrict ? (I.lower < x) : (I.lower <= x)
@inline checkbounds{T<:AbstractFloat}(I::UpperBound{T}, x::T) = I.ustrict ? (x < I.upper) : (x <= I.upper)
@inline function checkbounds{T<:AbstractFloat}(I::Interval{T}, x::T)
    if I.lstrict
        I.ustrict ? (I.lower < x < I.upper) : (I.lower < x <= I.upper)
    else
        I.ustrict ? (I.lower <= x < I.upper) : (I.lower <= x <  I.upper)
    end
end

type HyperValue{T<:Real}
    v::T
end
HyperValue{T<:Real}(v::T) = HyperValue{T}(v)

immutable HyperParameter{T<:Real}
    sym::Symbol
    fixed::Bool
    value::HyperValue{T}
    bound::Bounds{T}
end
function HyperParameter{T<:Real}(sym::Symbol, fixed::Bool, value::HyperValue{T}, bounds::Bounds{T})
    HyperParameter{T}(sym, fixed, value, bounds)
end
function HyperParameter{T<:Real}(sym::Symbol, fixed::Bool, value::T, bounds::Bounds{T})
    HyperParameter{T}(sym, fixed, HyperValue(value), bounds)
end
function HyperParameter{T<:Real}(sym::Symbol, fixed::Bool, value::T)
    HyperParameter{T}(sym, fixed, HyperValue(value), NullBound{T}())
end


function show(io::IO, θ::HyperParameter)
    print(io, string(θ.sym) * " = " * string(θ.value.v) * " ∈ " * boundstring(θ.bound))
end

#function ℝ{T<:AbstractFloat}

#=

abstract Kernel{T<:AbstractFloat}

immutable TestKernel1{T<:AbstractFloat} <: Kernel{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    function TestKernel1(α::T, γ::T)
        new(HyperParameter{T}(:α, HyperValue{T}(α), LowerBound{T}(true, zero(T))),
            HyperParameter{T}(:γ, HyperValue{T}(γ), Interval{T}(true, zero(T), false, one(T))))
    end
end
TestKernel1{T<:AbstractFloat}(α::T, γ::T) = TestKernel1{T}(α, γ)

@inline checkcase(κ::TestKernel1) = κ.gamma.value.v == 1 ? :γ1 : :∅

@inline phi{T<:AbstractFloat}(κ::TestKernel1{T},::Type,x::T)      = exp(κ.alpha.value.v*x^κ.gamma.value.v)
@inline phi{T<:AbstractFloat}(κ::TestKernel1{T},::Type{Val{:γ1}},x::T) = exp(κ.alpha.value.v*x)

kernel{T<:AbstractFloat}(κ::TestKernel1{T}, x::T) = phi(κ, Val{:∅}, x) # Val{checkcase(κ)}, x)


immutable TestKernel2{T<:AbstractFloat} <: Kernel{T}
    alpha::T
    gamma::T
    function TestKernel2(α::T, γ::T)
        new(α, γ)
    end
end
TestKernel2{T<:AbstractFloat}(α::T, γ::T) = TestKernel2{T}(α, γ)

kernel{T<:AbstractFloat}(κ::TestKernel2{T}, z::T) = exp(κ.alpha*z^κ.gamma)


function applykernel!{T<:AbstractFloat}(κ::Kernel{T}, X::Matrix{T})
    n, p = size(X)
    for j = 1:p, i = 1:n
        X[i,j] = kernel(κ, X[i,j])
    end
    X
end

=#
