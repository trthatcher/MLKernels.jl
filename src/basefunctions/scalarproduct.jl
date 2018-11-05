@doc raw"""
    ScalarProduct()
    
The scalar product is an inner product of the form:

```math
f(\mathbf{x}, \mathbf{y}) = \mathbf{x}^{\intercal}\mathbf{y}
```
"""
struct ScalarProduct <: InnerProduct end
@inline base_aggregate(::ScalarProduct, s::T, x::T, y::T) where {T} = s + x*y