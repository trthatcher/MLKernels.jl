abstract Bounds{T<:Real}


#====================
  Hypervalue Object
====================#

type ParameterValue{T<:Real}
    v::T
end
ParameterValue{T<:Real}(v::T) = ParameterValue{T}(v)

@inline *(a::Real, v::ParameterValue) = *(a, v.v)
@inline *(v::ParameterValue, a::Real) = *(v.v, a)

@inline +(a::Real, v::ParameterValue) = +(a, v.v)
@inline +(v::ParameterValue, a::Real) = +(v.v, a)

@inline ^(a::Real, v::ParameterValue)          = ^(a, v.v)
#@inline ^(v::ParameterValue, a::Integer)       = ^(v.v, a)
#@inline ^(v::ParameterValue, a::AbstractFloat) = ^(v.v, a)

@inline exp(v::ParameterValue)  = exp(v.v)
@inline tanh(v::ParameterValue) = tanh(v.v)


#=================
  Bounds Objects
=================#

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
    function Interval(lstrict, lower, ustrict, upper)
        if (lower > upper) || ((lstrict || ustrict) && lower == upper)
            error("Invalid bounds.")
        end
        new(lstrict, lower, ustrict, upper)
    end
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

@inline checkbounds{T<:Real}(I::NullBound{T}, x::T) = true
@inline function checkbounds{T<:Real}(I::LowerBound{T}, x::T)
    I.lstrict ? (I.lower < x) : (I.lower <= x)
end
@inline function checkbounds{T<:Real}(I::UpperBound{T}, x::T)
    I.ustrict ? (x < I.upper) : (x <= I.upper)
end
@inline function checkbounds{T<:Real}(I::Interval{T}, x::T)
    if I.lstrict
        I.ustrict ? (I.lower < x < I.upper) : (I.lower < x <= I.upper)
    else
        I.ustrict ? (I.lower <= x < I.upper) : (I.lower <= x <  I.upper)
    end
end

checkbounds{T<:Real}(I::Bounds{T}, x::ParameterValue{T}) = checkbounds(I, x.v)

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

immutable Parameter{T<:Real}
    sym::Symbol
    fixed::Bool
    val::ParameterValue{T}
    bound::Bounds{T}
    function Parameter(sym::Symbol, fixed::Bool, value::ParameterValue{T}, bound::Bounds{T})
        if !checkbounds(bound, value)
            error("$(sym) = $(value.v) must be in range " * boundstring(bound))
        end
        new(sym, fixed, value, bound)
    end
end
function Parameter{T<:Real}(sym::Symbol, fixed::Bool, value::ParameterValue{T}, bounds::Bounds{T})
    Parameter{T}(sym, fixed, value, bounds)
end
function Parameter{T<:Real}(sym::Symbol, fixed::Bool, value::T, bounds::Bounds{T})
    Parameter{T}(sym, fixed, ParameterValue(value), bounds)
end
function Parameter{T<:Real}(sym::Symbol, fixed::Bool, value::T)
    Parameter{T}(sym, fixed, ParameterValue(value), NullBound{T}())
end

function show(io::IO, θ::Parameter)
    print(io, string(θ.sym) * " = " * string(θ.val.v) * " ∈ " * boundstring(θ.bound))
end

valstring(θ::Parameter) = string(θ.sym) * "=" * string(θ.val.v)


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
