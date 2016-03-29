#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module MLKernels

import Base: call, convert, eltype, print, show, string, ==, *, /, +, -, ^, besselk, exp, gamma, 
             tanh

export
    # Functions
    ∘,
    kernel,
    kernelmatrix,
    centerkernelmatrix!,
    centerkernelmatrix,
    nystrom,

    ismercer,
    isnegdef,
    ispositive,
    isnonnegative,

    # Hyper Parameters
    Bound,
    NullBound,
    LowerBound,
    UpperBound,
    Interval,
    Parameter,
    Value,
    Variable,
    Fixed,

    # Kernel Type
    Kernel,
        StandardKernel,
            PairwiseKernel,
                AdditiveKernel,
                    SquaredDistanceKernel,
                    SineSquaredKernel,
                    ChiSquaredKernel,
                    ScalarProductKernel,
                ARD,
            KernelComposition,
        KernelOperation,
            KernelAffinity,
            KernelProduct,
            KernelSum,
    
    # Composition Classes
    CompositionClass,
        PositiveMercerClass,
            ExponentialClass,
            GammaExponentialClass,
            RationalClass,
            GammaRationalClass,
            MaternClass,
            ExponentiatedClass,
        PolynomialClass,
        NonNegativeNegativeDefiniteClass,
            PowerClass,
            LogClass,
            GammaLogClass,
        SigmoidClass,

    
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

include("meta.jl")
include("type_Parameter.jl")
include("type_Kernel.jl")
include("function_pairwise.jl")
include("function_kernel.jl")
#include("kernelapproximation.jl")

end # MLKernels
