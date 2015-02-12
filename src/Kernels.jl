#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module Kernels

import Base: show

export
	# Mercer Kernels
	MercerKernel,
	LinearKernel,
	PolynomialKernel,
	GaussianKernel,
	ExponentialKernel,
	SigmoidKernel,
	RationalQuadraticKernel,
	MultiQuadraticKernel,
	InverseMultiQuadraticKernel,
	PowerKernel,
	LogKernel,
	SplineKernel
	# Kernel Matrix Functions
	#center_kernelmatrix!,
	#center_kernelmatrix,
	#kernelmatrix

include("MercerKernels.jl")
#include("Gramian.jl")

end # Kernels
