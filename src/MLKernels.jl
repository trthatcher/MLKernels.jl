#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module MLKernels

import Base: show, eltype, convert, promote #, call

export
    # Functions
    description,
    isposdef_kernel,
    iscondposdef_kernel,
    kernel,
    kernelparameters,
    kernelmatrix,
    center_kernelmatrix!,
    center_kernelmatrix,
    nystrom,

    # Kernel Types
    Kernel,
        SimpleKernel,
            StandardKernel,
                SquaredDistanceKernel,
                    ExponentialKernel,
                    RationalQuadraticKernel,
                    PowerKernel,
                    LogKernel,
                ScalarProductKernel,
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
include("auxfunctions.jl")
include("kernels.jl")
include("kernelmatrix.jl")
#include("kernelmatrixapprox.jl")

end # MLKernels
