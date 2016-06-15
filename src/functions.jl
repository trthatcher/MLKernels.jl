#===================================================================================================
  RealFunctions
===================================================================================================#

abstract MathematicalFunction{T}

abstract RealFunction{T<:AbstractFloat} <: MathematicalFunction{T}

eltype{T}(::RealFunction{T}) = T 

doc"`ismercer(f)`: returns `true` if `f` is a valid real-valued Mercer kernel"
ismercer(::RealFunction) = false

doc"`isnegdef(f)`: returns `true` if `f` is a valid real-valued negative-definite kernel"
isnegdef(::RealFunction) = false

doc"`ismetric(f)`: returns `true` if `f` is a valid metric"
ismetric(::RealFunction) = false

doc"`isinnerprod(f)`: returns `true` if `f` is a valid inner product"
isinnerprod(::RealFunction) = false

doc"`attainszero(f)`: returns `true` if ∃x,y,θ such that f(x,y;θ) = 0"
attainszero(::RealFunction) = true

doc"`attainspositive(f)`: returns `true` if ∃x,y,θ such that f(x,y;θ) > 0"
attainspositive(::RealFunction) = true

doc"`attainsnegative(f)`: returns `true` if ∃x,y,θ such that f(x,y;θ) < 0"
attainsnegative(::RealFunction) = true

doc"`isnonnegative(f)`: returns `true` if f(x,y;θ) > 0 ∀x,y,θ"
isnonnegative(f::RealFunction) = !attainsnegative(f)

doc"`ispositive(f)`: returns `true` if f(x,y;θ) ≧ 0 ∀x,y,θ"
ispositive(f::RealFunction) = !attainsnegative(f) && !attainszero(f) &&  attainspositive(f)

doc"`isnegative(f)`: returns `true` if f(x,y;θ) ≦ 0 ∀x,y,θ"
isnegative(k::RealFunction) =  attainsnegative(f) && !attainszero(f) && !attainspositive(f)

function show(io::IO, f::RealFunction)
    print(io, description_string(f))
end

function convert{T<:AbstractFloat,F<:RealFunction}(::Type{RealFunction{T}}, f::F)
    convert(F.name.primary{T}, f)
end

# Functions

include("functions/pairwisefunction.jl")
include("functions/compositefunction.jl")
#include("pointwisefunction.jl")
