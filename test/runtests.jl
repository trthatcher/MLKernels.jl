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

const FloatingPointTypes = (Float32, Float64)
const IntegerTypes = (Int32, Int64)
const MLK = MLKernels

include("reference.jl")

@testset "Testing utility functions" begin
    include("utils.jl")
end

@testset "Testing BaseFunction types" begin
    include("basefunctions.jl")
end

@testset "Testing BaseFunction scalar and matrix evaluation" begin
    include("basematrix.jl")
end

@testset "Testing Kernel types" begin
    include("kernelfunctions.jl")
end
#include("kernelmatrix.jl")
#include("nystrom.jl")