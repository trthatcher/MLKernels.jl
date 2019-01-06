# Pairwise Functions =======================================================================

abstract type BaseFunction end

@inline isstationary(::BaseFunction) = false
@inline isisotropic(::BaseFunction) = false


# Inner Products ===========================================================================

abstract type InnerProduct <: BaseFunction end

const inner_products = [
    "scalarproduct"
]

for fname in inner_products
    include(joinpath("basefunctions", "$(fname).jl"))
end


# Pre-Metrics ==============================================================================

abstract type PreMetric <: BaseFunction end


# Metrics ==================================================================================

abstract type Metric <: PreMetric end

const metrics = [
    "squaredeuclidean"
]

for fname in metrics
    include(joinpath("basefunctions", "$(fname).jl"))
end