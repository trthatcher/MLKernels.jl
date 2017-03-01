#===================================================================================================
  Metrics - Consume two vectors
===================================================================================================#

abstract PairwiseFunction

@inline pairwise_initiate{T}(::PairwiseFunction, ::Type{T}) = zero(T)
@inline pairwise_return{T}(::PairwiseFunction, s::T) = s



abstract InnerProduct <: PairwiseFunction

doc"ScalarProduct() = xᵀy"
immutable ScalarProduct <: InnerProduct end
@inline pairwise_aggregate{T}(f::ScalarProduct, s::T, x::T, y::T) = s + x*y



abstract PreMetric <: PairwiseFunction

doc"ChiSquared() = Σⱼ(xⱼ-yⱼ)²/(xⱼ+yⱼ)"
immutable ChiSquared <: PreMetric end
@inline function pairwise_aggregate{T}(::ChiSquared, s::T, x::T, y::T)
    x == y == zero(T) ? s : s + (x-y)^2/(x+y)
end

doc"SineSquared(p) = Σⱼsin²(xⱼ-yⱼ)"
immutable SineSquared <: PreMetric end
@inline pairwise_aggregate{T}(f::SineSquared, s::T, x::T, y::T) = s + sin(x-y)^2



abstract Metric <: PreMetric

doc"SquaredEuclidean() = (x-y)ᵀ(x-y)"
immutable SquaredEuclidean <: PreMetric end
@inline pairwise_aggregate{T}(f::SquaredEuclidean, s::T, x::T, y::T) = s + (x-y)^2
