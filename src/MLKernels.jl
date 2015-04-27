#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module MLKernels

import Base: show, exp, eltype, isposdef, convert, promote #, call

export
    # Functions
    description,
    isposdef_kernel,
    kernel,
    kernelmatrix,
    gramian_matrix,
    lagged_gramian_matrix,
    center_kernelmatrix!,
    center_kernelmatrix,
    nystrom,
    parameters,
    # Types
    Kernel,
        SimpleKernel,
            StandardKernel,
                # PointwiseProductKernel,
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

include("kernels.jl")
include("kernelmatrix.jl")

end # MLKernels
