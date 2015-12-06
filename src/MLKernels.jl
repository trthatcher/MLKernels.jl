#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module MLKernels

import Base: call, convert, eltype, show, *, +, ^, exp, <|, |>

export
    # Functions
    ∘,
    ismercer,
    isnegdef,
    attainszero,
    ispositive,
    isnonnegative,
    isnonpositive,
    isnegative,
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
                SeparableKernel,
                    ScalarProductKernel,
                    MercerSigmoidKernel,
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
include("kernelfunctions.jl")
include("kernelapproximation.jl")

end # MLKernels
