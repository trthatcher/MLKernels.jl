# Unit Tests
using Test
using MLKernels

using MLKernels:
    BaseFunction,
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
include("basefunctions.jl")
include("basematrix.jl")

include("kernelfunctions.jl")
include("kernelmatrix.jl")
include("nystrom.jl")