#===================================================================================================
  Kernel Kernels Module
===================================================================================================#

module MLKernels

import Base: call, convert, eltype, print, show, string, ==, *, /, +, -, ^, besselk, exp, gamma, 
             tanh

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

    # Memory
    MemoryLayout,
    RowMajor,
    ColumnMajor,

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
    isnegdef,

    # Kernel Matrix
    kernel,
    kernelmatrix,
    kernelmatrix!,

    # Kernel Approximation
    NystromFact,
    nystrom

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

# Row major and column major ordering are supported
abstract MemoryLayout

immutable ColumnMajor <: MemoryLayout end
immutable RowMajor    <: MemoryLayout end


include("common.jl")
include("pairwise.jl")
include("pairwisematrix.jl")
include("kernel.jl")
include("kernelmatrix.jl")
include("kernelmatrixapproximation.jl")
    
end # MLKernels
