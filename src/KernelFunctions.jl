#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module KernelFunctions

import Base: show, call, exp, eltype, isposdef, convert, promote

export
    # Functions
    arguments,
    description,
    isposdef_kernel,
    is_euclidean_distance,
    is_scalar_product,
    kernel_function,
    scalar_kernel_function!,
    scalar_kernel_function,
    # Types
    Kernel,
        SimpleKernel,
            StandardKernel,
                # PointwiseProductKernel,
                EuclideanDistanceKernel,
	                GaussianKernel,
	                LaplacianKernel,
            	    RationalQuadraticKernel,
            	    MultiQuadraticKernel,
            	    InverseMultiQuadraticKernel,
            	    PowerKernel,
            	    LogKernel,
                ScalarProductKernel,
	                LinearKernel,
	                PolynomialKernel,
            	    SigmoidKernel,
            ScaledKernel,
        CompositeKernel,
            KernelProduct,
            KernelSum

include("kerneltypes.jl")  # General and composite kernels

#include("kernelmatrix.jl")

end # KernelFunctions
