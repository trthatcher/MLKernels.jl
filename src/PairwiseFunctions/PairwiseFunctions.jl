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

    # Memory
    MemoryLayout,
    RowMajor,
    ColumnMajor,

    # Pairwise function evaluation
    pairwise,
    pairwisematrix,
    pairwisematrix!

import LinearAlgebra

abstract type MemoryLayout end

struct ColumnMajor <: MemoryLayout end
struct RowMajor    <: MemoryLayout end

include("common.jl")
include("pairwise.jl")
include("pairwisematrix.jl")

end # MLKernels
