#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module Kernels

import Base: show, call, exp

export
    # Functions
    kernelfunction,
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

include("KernelFunctions.jl")

end # Kernels
