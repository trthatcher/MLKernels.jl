# Unit Tests
using Base.Test
using MLKernels

using MLKernels:
    PairwiseFunction,
        InnerProduct,
            ScalarProduct,
        PreMetric,
            ChiSquared,
            SineSquared,
            Metric,
                SquaredEuclidean

FloatingPointTypes = (Float32, Float64)
IntegerTypes = (Int32, UInt32, Int64, UInt64)
MOD = MLKernels

include("hyperparameter.jl")
include("common.jl")

include("reference.jl")
include("pairwise.jl")
include("pairwisematrix.jl")
include("kernel.jl")
include("kernelmatrix.jl")
include("kernelmatrixapproximation.jl")
