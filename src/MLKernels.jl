#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module MLKernels

import Base: show, exp, eltype, isposdef, convert, promote #, call

export
    # Functions
    description,
    kernel,
    kernel_dx,
    kernel_dy,
    kernel_dp,
    kernel_dw,
    kernel_dxdy,
    kernelmatrix,
    kernelmatrix_dx,
    kernelmatrix_dy,
    kernelmatrix_dp,
    kernelmatrix_dxdy,
    scprodmatrix,
    sqdistmatrix,
    center_kernelmatrix!,
    center_kernelmatrix,
    nystrom,

    # Types
    Kernel,
        SimpleKernel,
            StandardKernel,
                SquaredDistanceKernel,
                    GaussianKernel, SquaredExponentialKernel,
                    LaplacianKernel, ExponentialKernel,
                    RationalQuadraticKernel,
                    MultiQuadraticKernel,
                    InverseMultiQuadraticKernel,
                    PowerKernel,
                    LogKernel,
                    PeriodicKernel,
                ScalarProductKernel,
                    LinearKernel,
                    PolynomialKernel,
                    SigmoidKernel,
                SeparableKernel,
                    MercerSigmoidKernel,
                ARD,
            ScaledKernel,
        CompositeKernel,
            KernelProduct,
            KernelSum

include("meta.jl")
include("vectorfunctions.jl")
include("matrixfunctions.jl")
include("kernels.jl")
include("kernelmatrix.jl")
include("kernelapprox.jl")
include("kernelmatrixderiv.jl")

end # MLKernels
