#===================================================================================================
  Kernel Kernels Module
===================================================================================================#

module MLKernels

import Base: convert, eltype, print, show, string, ==, *, /, +, -, ^, exp, tanh

export

    # Hyper Parameters
    Bound,
        OpenBound,
        ClosedBound,
        NullBound,

    Interval,
    interval,

    HyperParameter,
    getvalue,
    setvalue!,
    checkvalue,
    gettheta,
    settheta!,
    checktheta,

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

    # Kernel Function Properties
    ismercer,
    isnegdef,

    # Kernel Matrix
    kernel,
    kernelmatrix,
    kernelmatrix!,

    # Kernel Approximation
    NystromFact,
    nystrom


using SpecialFunctions: besselk, gamma

include("HyperParameters/HyperParameters.jl")
using MLKernels.HyperParameters:
    Bound,
        OpenBound,
        ClosedBound,
        NullBound,

    Interval,
    interval,

    HyperParameter,
    getvalue,
    setvalue!,
    checkvalue,
    gettheta,
    checktheta,
    settheta!,
    lowerboundtheta,
    upperboundtheta

import MLKernels.HyperParameters: gettheta, checktheta, settheta!

include("PairwiseFunctions/PairwiseFunctions.jl")
using MLKernels.PairwiseFunctions:

    # Pairwise Functions
    PairwiseFunction,
        InnerProduct,
            ScalarProduct,
        PreMetric,
            ChiSquared,
            SineSquared,
            Metric,
                SquaredEuclidean,

    # Memory
    MemoryLayout,
    RowMajor,
    ColumnMajor,

    # Pairwise function evaluation
    pairwise,
    pairwisematrix,
    pairwisematrix!

import MLKernels.PairwiseFunctions: isstationary, isisotropic

include("kernel.jl")
include("kernelmatrix.jl")
include("kernelmatrixapproximation.jl")

end # MLKernels
