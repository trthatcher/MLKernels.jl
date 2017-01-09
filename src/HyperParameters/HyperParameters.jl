module HyperParameters

import Base: convert, eltype, show, ==, *, /, +, -, ^, besselk, exp, gamma, tanh

export 
    Bound,
    Interval,
    leftbounded,
    rightbounded,
    unbounded,
    checkbounds,
    Variable,
    fixed, 
    Argument,
    HyperParameter

include("bound.jl")
include("interval.jl")
include("hyperparameter.jl")

end # End HyperParameter
