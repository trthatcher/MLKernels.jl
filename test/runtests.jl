# Unit Tests
using Base.Test
importall MLKernels

include("utilities.jl")

indent_block = "   "

include("test_common.jl")
include("test_kernels.jl")
include("test_pairwise.jl")
include("test_kernelfunctions.jl")
include("test_kernelapproximation.jl")
