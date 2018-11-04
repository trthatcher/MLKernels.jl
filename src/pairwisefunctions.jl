# Pairwise Functions =======================================================================

abstract type PairwiseFunction end

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


# Inner Products ===========================================================================

abstract type InnerProduct <: PairwiseFunction end

const inner_products = [
    "scalarproduct"
]

for fname in inner_products
    include(joinpath("pairwise", "$(fname).jl"))
end


# Pre-Metrics ==============================================================================

abstract type PreMetric <: PairwiseFunction end

const pre_metrics = [
    "chisquared",
    "sinesquared"
]

for fname in pre_metrics
    include(joinpath("pairwise", "$(fname).jl"))
end


# Metrics ==================================================================================

abstract type Metric <: PreMetric end

const metrics = [
    "squaredeuclidean"
]

for fname in metrics
    include(joinpath("pairwise", "$(fname).jl"))
end
