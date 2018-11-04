@doc raw"""
    SineSquared

The Sine-Squared pairwise function is given by:

```math
f(\mathbf{x}, \mathbf{y}) = \sine^2 \left(x_i - y_i\right)
``` 
"""
struct SineSquared <: PreMetric end
@inline pairwise_aggregate(::SineSquared, s::T, x::T, y::T) where {T} = s + sin(x-y)^2
@inline isstationary(::SineSquared) = true