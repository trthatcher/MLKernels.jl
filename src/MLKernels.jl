#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module MLKernels

import Base: call, convert, eltype, show, *, +, ^, exp, tanh, <|, |>

export
    # Functions
    ∘,
    ismercer,
    isnegdef,
    kernel,
    kernelmatrix,
    centerkernelmatrix!,
    centerkernelmatrix,
    nystrom,

    # Kernel Type
    Kernel,
        # Subtypes
        BaseKernel,
            AdditiveKernel,
                SquaredDistanceKernel,
                SineSquaredKernel,
                ChiSquaredKernel,
                ScalarProductKernel,
            ARD,
        KernelComposition,
        KernelOperation,
            KernelProduct,
            KernelSum,
        # Composition Classes
        CompositionClass,
            ExponentialClass,
            RationalQuadraticClass,
            PolynomialClass,
            MaternClass,
            ExponentiatedClass,
            SigmoidClass,
            PowerClass,
            LogClass,
        # Kernel Constructors
        GaussianKernel,
            SquaredExponentialKernel,
            RadialBasisKernel,
        LaplacianKernel,
        PeriodicKernel,
        RationalQuadraticKernel,
        MaternKernel,
            MatérnKernel,
        PolynomialKernel,
        LinearKernel,
        SigmoidKernel


include("common.jl")
include("meta.jl")
include("kernels.jl")
include("pairwise.jl")
#include("kernelfunctions.jl")
#include("kernelapproximation.jl")

end # MLKernels
