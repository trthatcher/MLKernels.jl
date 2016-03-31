# Unit Tests
using Base.Test
importall MLKernels

FloatingPointTypes = (Float32, Float64)
IntegerTypes = (Int32, UInt32, Int64, UInt64)
MOD = MLKernels

include("reference.jl")

include("test_type_HyperParameter.jl")
include("test_type_PairwiseKernel.jl")
include("test_type_CompositionClass.jl")
include("test_type_KernelComposition.jl")
include("test_type_KernelAffinity.jl")
include("test_pairwise.jl")
#include("test_kernelfunctions.jl")
#include("test_kernelapproximation.jl")
