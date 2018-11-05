@doc raw"""
    ChiSquared()

The Chi-Squared base function is given by:

```math
f(\mathbf{x}, \mathbf{y}) = \sum_i \frac{(x_i - y_i)^2}{x_i + y_i}
```
"""
struct ChiSquared <: PreMetric end

@inline function base_aggregate(::ChiSquared, s::T, x::T, y::T) where {T}
    x == y == zero(T) ? s : s + (x-y)^2/(x+y)
end
