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
    # Abstract Types
    MercerKernel,
    SimpleMercerKernel,
    CompositeMercerKernel,
    TransformedMercerKernel,
    # Other types
    ScalableMercerKernel,
    #Transformed Mercer Kernels
    ScaledMercerKernel,
    ExponentialMercerKernel,
    ExponentiatedMercerKernel,
    # Composite Mercer Kernels
    MercerKernelProduct,
    MercerKernelSum,
	# Mercer Kernels
    PointwiseProductKernel,
    GenericKernel,
	LinearKernel,
	PolynomialKernel,
	GaussianKernel,
	LaplacianKernel,
	SigmoidKernel,
	RationalQuadraticKernel,
	MultiQuadraticKernel,
	InverseMultiQuadraticKernel,
	PowerKernel,
	LogKernel,
	SplineKernel

include("MercerKernels.jl")
#include("Gramian.jl")

end # Kernels
