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
MLK = MLKernels
HP = MLKernels.HyperParameters

include("reference.jl")

include("HyperParameters/hyperparameter.jl")

include("utils.jl")
include("pairwisefunctions.jl")
include("pairwisematrix.jl")

include("kernelfunctions.jl")
include("kernelmatrix.jl")
include("nystrom.jl")
