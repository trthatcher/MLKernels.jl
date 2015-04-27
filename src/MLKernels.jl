#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module MLKernels

import Base: show, exp, eltype, isposdef, convert, promote #, call

export
    # Functions
    description,
    kernel,
    dkernel_dx,
    dkernel_dy,
    dkernel_dp,
    d2kernel_dxdy,
    kernelmatrix,
    gramian_matrix,
    lagged_gramian_matrix,
    center_kernelmatrix!,
    center_kernelmatrix,

    # Types
    Kernel,
        SimpleKernel,
            StandardKernel,
                EuclideanDistanceKernel,
	                GaussianKernel, SquaredExponentialKernel,
	                LaplacianKernel, ExponentialKernel,
            	    RationalQuadraticKernel,
            	    MultiQuadraticKernel,
            	    InverseMultiQuadraticKernel,
            	    PowerKernel,
            	    LogKernel,
                ScalarProductKernel,
	                LinearKernel,
	                PolynomialKernel,
            	    SigmoidKernel,
                SeparableKernel,
                    MercerSigmoidKernel,
            ScaledKernel,
        CompositeKernel,
            KernelProduct,
            KernelSum

include("matrixfunctions.jl")
include("kernels.jl")
include("kernelmatrix.jl")
include("kernelapprox.jl")

end # MLKernels
