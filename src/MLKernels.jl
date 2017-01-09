#===================================================================================================
  Kernel Kernels Module
===================================================================================================#

module MLKernels

import Base: call, convert, eltype, print, show, string, ==, *, /, +, -, ^, besselk, exp, gamma, 
             tanh

export

    # Hyper Parameters
    Bound,
    Interval,
    leftbounded,
    rightbounded,
    unbounded,
    checkbounds,
    Variable,
    fixed,
    Argument,
    HyperParameter,

    # Memory
    MemoryLayout,
    RowMajor,
    ColumnMajor,

    # Pairwise Function Type
    PairwiseFunction,
        InnerProduct,
            ScalarProduct,
        PreMetric,
            ChiSquared,
            SineSquared,
            Metric,
                Euclidean,
                SquaredEuclidean,

    # Pairwise Matrix
    pairwise,
    pairwisematrix,
    pairwisematrix!,


    # Kernel Function Type
    Kernel,
        MercerKernel, 
            ExponentialKernel,
                LaplacianKernel,
            SquaredExponentialKernel,
                GaussianKernel,
                RadialBasisKernel,
            GammaExponentialKernel,
            RationalQuadraticKernel,
            GammaRationalKernel,
            MaternKernel,
            LinearKernel,
            PolynomialKernel,
            ExponentiatedKernel,
            PeriodicKernel,
        NegativeDefiniteKernel,
            PowerKernel,
            LogKernel,
        SigmoidKernel,

    # Kernel Functions
    ismercer,

    # Kernel Matrix
    kernel,
    kernelmatrix,
    kernelmatrix!

include("HyperParameters/HyperParameters.jl")
using MLKernels.HyperParameters: 
    Bound,
    Interval,
    leftbounded,
    rightbounded,
    unbounded,
    checkbounds,
    HyperParameter


# Row major and column major ordering are supported
abstract MemoryLayout

immutable ColumnMajor <: MemoryLayout end
immutable RowMajor    <: MemoryLayout end


include("common.jl")
include("pairwisefunction.jl")
include("pairwisematrix.jl")
include("kernelfunction.jl")
include("kernelmatrix.jl")
    
end # MLKernels
