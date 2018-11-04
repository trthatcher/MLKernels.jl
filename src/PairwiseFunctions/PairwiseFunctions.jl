#===================================================================================================
  Kernel Kernels Module
===================================================================================================#

module PairwiseFunctions

export

    # Pairwise Functions
    PairwiseFunction,
        InnerProduct,
            ScalarProduct,
        PreMetric,
            ChiSquared,
            SineSquared,
            Metric,
                SquaredEuclidean,

    # Pairwise Function Properties
    isstationary,
    isisotropic,

    # Orientation
    Orientation,

    # Pairwise function evaluation
    pairwise,
    pairwisematrix,
    pairwisematrix!

import LinearAlgebra

@doc raw"""
    Orientation

Union of the two `Val` types representing the data matrix orientations:

  1. `Val{:row}` identifies when observation vector corresponds to a row of the data matrix
  2. `Val{:col}` identifies when each observation vector corresponds to a column of the data
     matrix
"""
const Orientation = Union{Val{:row}, Val{:col}}

include("common.jl")
include("pairwise.jl")
include("pairwisematrix.jl")

end # PairwiseFunctions
