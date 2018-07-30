# Unit Tests
using Test
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

using SpecialFunctions:
    besselk, gamma

import LinearAlgebra
import Statistics

FloatingPointTypes = (Float32, Float64)
IntegerTypes = (Int32, Int64)
MOD = MLKernels
MODHP = MLKernels.HyperParameters
MODPF = MLKernels.PairwiseFunctions

include("reference.jl")

include("HyperParameters/hyperparameter.jl")

include("PairwiseFunctions/common.jl")
include("PairwiseFunctions/pairwise.jl")
include("PairwiseFunctions/pairwisematrix.jl")

include("kernel.jl")
include("kernelmatrix.jl")
include("kernelmatrixapproximation.jl")
