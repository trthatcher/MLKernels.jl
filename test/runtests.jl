# Unit Tests
using Base.Test
importall MLKernels

FloatingPointTypes = (Float32, Float64)
MOD = MLKernels

include("reference.jl")

include("test_type_PairwiseKernel.jl")
include("test_type_CompositionClass.jl")
include("test_type_CompositionKernel.jl")
include("test_pairwise.jl")
#include("test_kernelfunctions.jl")
#include("test_kernelapproximation.jl")
