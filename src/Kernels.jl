#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module KERNEL

using MATRIX

import Base: show

export
	# Mercer Kernels
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
	SplineKernel,
	# Kernel Matrix Functions
	center_kernelmatrix!,
	center_kernelmatrix,
	kernelmatrix


end # KERNEL
