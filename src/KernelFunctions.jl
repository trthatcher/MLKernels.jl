#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module KernelFunctions

import Base: show, call, exp, isposdef

export
    # Functions
    arguments,
    description,
    isposdef_kernel,
    kernel_function,
    # Types
    Kernel,
        SimpleKernel,
            StandardKernel,
                PointwiseProductKernel,
                GenericKernel,
                StationaryKernel,
	                GaussianKernel,
	                LaplacianKernel,
            	    RationalQuadraticKernel,
            	    MultiQuadraticKernel,
            	    InverseMultiQuadraticKernel,
            	    PowerKernel,
            	    LogKernel,
                NonStationaryKernel,
	                LinearKernel,
	                PolynomialKernel,
            	    SigmoidKernel,
            TransformedKernel,
                ExponentialKernel,
                ExponentiatedKernel,
            ScalableKernel,
            ScaledKernel,
        CompositeKernel,
            KernelProduct,
            KernelSum

include("Kernels.jl")

end # KernelFunctions
