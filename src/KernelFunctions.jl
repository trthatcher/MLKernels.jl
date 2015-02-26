#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module KernelFunctions

import Base: show, call, exp

export
    # Functions
    kernel_function,
    arguments,
    description,
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
