@doc raw"""
    ChiSquared()

The Chi-Squared pairwise function is given by:

```math
f(\mathbf{x}, \mathbf{y}) = \sum_i \frac{(x_i - y_i)^2}{x_i + y_i}
```
"""
struct ChiSquared <: PreMetric end

@inline function pairwise_aggregate(::ChiSquared, s::T, x::T, y::T) where {T}
    x == y == zero(T) ? s : s + (x-y)^2/(x+y)
end
