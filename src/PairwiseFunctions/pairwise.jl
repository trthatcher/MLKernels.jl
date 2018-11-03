#===================================================================================================
  Metrics - Consume two vectors
===================================================================================================#

abstract type PairwiseFunction end

@inline pairwise_initiate(::PairwiseFunction, ::Type{T}) where {T} = zero(T)
@inline pairwise_return(::PairwiseFunction, s::T) where {T} = s

"""
    isstationary(κ::Kernel)

Returns `true` if the kernel `κ` is a stationary kernel; `false` otherwise.
"""
@inline isstationary(::PairwiseFunction) = false

"""
    isisotropic(κ::Kernel)

Returns `true` if the kernel `κ` is an isotropic kernel; `false` otherwise.
"""
@inline isisotropic(::PairwiseFunction) = false



abstract type InnerProduct <: PairwiseFunction end

"ScalarProduct() = xᵀy"
struct ScalarProduct <: InnerProduct end
@inline pairwise_aggregate(::ScalarProduct, s::T, x::T, y::T) where {T} = s + x*y



abstract type PreMetric <: PairwiseFunction end

"ChiSquared() = Σⱼ(xⱼ-yⱼ)²/(xⱼ+yⱼ)"
struct ChiSquared <: PreMetric end
@inline function pairwise_aggregate(::ChiSquared, s::T, x::T, y::T) where {T}
    x == y == zero(T) ? s : s + (x-y)^2/(x+y)
end

"SineSquared(p) = Σⱼsin²(xⱼ-yⱼ)"
struct SineSquared <: PreMetric end
@inline pairwise_aggregate(::SineSquared, s::T, x::T, y::T) where {T} = s + sin(x-y)^2
@inline isstationary(::SineSquared) = true


abstract type Metric <: PreMetric end

"SquaredEuclidean() = (x-y)ᵀ(x-y)"
struct SquaredEuclidean <: PreMetric end
@inline pairwise_aggregate(::SquaredEuclidean, s::T, x::T, y::T) where {T} = s + (x-y)^2
@inline isstationary(::SquaredEuclidean) = true
@inline isisotropic(::SquaredEuclidean)  = true
