#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module MLKernels

import Base: show, eltype, convert, promote #, call

export
    # Functions
    description,
    ismercer,
    kernel,
    kernelparameters,
    kernel_dx,
    kernel_dy,
    kernel_dxdy,
    kernel_dp,
    kernelmatrix,
    kernelmatrix_dx,
    kernelmatrix_dy,
    kernelmatrix_dxdy,
    kernelmatrix_dp,
    center_kernelmatrix!,
    center_kernelmatrix,
    nystrom,

    # Types
    Kernel,
        SimpleKernel,
            StandardKernel,
                SquaredDistanceKernel,
                    ExponentialKernel,
                    RationalQuadraticKernel,
                    PowerKernel,
                    LogKernel,
                ScalarProductKernel,
                    LinearKernel,
                    PolynomialKernel,
                    SigmoidKernel,
                SeparableKernel,
                    MercerSigmoidKernel,
                PeriodicKernel,
            ARD,
        CompositeKernel,
            KernelProduct,
            KernelSum

include("meta.jl")
include("vectorfunctions.jl")
include("matrixfunctions.jl")
include("kernels.jl")
include("kernelderiv.jl")
include("kernelmatrix.jl")
include("kernelmatrixderiv.jl")
#include("kernelmatrixapprox.jl")

end # MLKernels
