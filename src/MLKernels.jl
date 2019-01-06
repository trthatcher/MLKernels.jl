#===================================================================================================
  Kernel Kernels Module
===================================================================================================#

module MLKernels

import Base: convert, eltype, print, show, string, ==, *, /, +, -, ^, exp, tanh

export

    # Memory
    Orientation,

    # Kernel Functions
    Kernel,
        MercerKernel,
            AbstractExponentialKernel,
                ExponentialKernel,
                LaplacianKernel,
                SquaredExponentialKernel,
                GaussianKernel,
                RadialBasisKernel,
                GammaExponentialKernel,
            AbstractRationalQuadraticKernel,
                RationalQuadraticKernel,
                GammaRationalQuadraticKernel,
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
    isstationary,
    isisotropic,

    # Kernel Matrix
    kernel,
    kernelmatrix,
    kernelmatrix!,
    centerkernelmatrix!,
    centerkernelmatrix,

    # Kernel Approximation
    NystromFact,
    nystrom


using SpecialFunctions: besselk, gamma
import LinearAlgebra
import Statistics

@doc raw"""
    Orientation

Union of the two `Val` types representing the data matrix orientations:

  1. `Val{:row}` identifies when observation vector corresponds to a row of the data matrix
  2. `Val{:col}` identifies when each observation vector corresponds to a column of the data
     matrix
"""
const Orientation = Union{Val{:row}, Val{:col}}

include("utils.jl")

include("basefunctions.jl")
include("basematrix.jl")

include("kernelfunctions.jl")
include("kernelmatrix.jl")
include("nystrom.jl")

include("deprecated.jl")

end # MLKernels