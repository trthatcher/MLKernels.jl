#===================================================================================================
  Metrics - Consume two vectors
===================================================================================================#

abstract type PairwiseFunction end

@inline pairwise_initiate{T}(::PairwiseFunction, ::Type{T}) = zero(T)
@inline pairwise_return{T}(::PairwiseFunction, s::T) = s

@inline isstationary(::PairwiseFunction) = false
@inline isisotropic(::PairwiseFunction) = false



abstract type InnerProduct <: PairwiseFunction end

doc"ScalarProduct() = xᵀy"
struct ScalarProduct <: InnerProduct end
@inline pairwise_aggregate{T}(f::ScalarProduct, s::T, x::T, y::T) = s + x*y



abstract type PreMetric <: PairwiseFunction end

doc"ChiSquared() = Σⱼ(xⱼ-yⱼ)²/(xⱼ+yⱼ)"
struct ChiSquared <: PreMetric end
@inline function pairwise_aggregate{T}(::ChiSquared, s::T, x::T, y::T)
    x == y == zero(T) ? s : s + (x-y)^2/(x+y)
end

doc"SineSquared(p) = Σⱼsin²(xⱼ-yⱼ)"
struct SineSquared <: PreMetric end
@inline pairwise_aggregate{T}(f::SineSquared, s::T, x::T, y::T) = s + sin(x-y)^2
@inline isstationary(f::SineSquared) = true


abstract type Metric <: PreMetric end

doc"SquaredEuclidean() = (x-y)ᵀ(x-y)"
struct SquaredEuclidean <: PreMetric end
@inline pairwise_aggregate{T}(f::SquaredEuclidean, s::T, x::T, y::T) = s + (x-y)^2
@inline isstationary(f::SquaredEuclidean) = true
@inline isisotropic(f::SquaredEuclidean)  = true
