#===================================================================================================
  Kernel Functions Module
===================================================================================================#

module MLKernels

import Base: call, convert, eltype, print, show, string, ==, *, /, +, -, ^, besselk, exp, gamma, 
             tanh

export
    # Functions
    ismercer,
    isnegdef,
    ismetric,
    isinnerprod,
    ispositive,
    isnonnegative,
    ∘,
    pairwise,
    pairwisematrix,

    # Hyper Parameters
    Bound,
    Interval,
    leftbounded,
    rightbounded,
    unbounded,
    checkbounds,
    Variable,
    fixed,
    Argument,
    HyperParameter,

    # Kernel Type
    MathematicalFunction,
        RealFunction,
            PairwiseFunction,
                SymmetricFunction,
                    Metric,
                        Euclidean,
                        SquaredEuclidean,
                        ChiSquared,
                    InnerProduct,
                        ScalarProduct,
                    SineSquaredKernel,
            CompositeFunction,
            PointwiseFunction,
                AffineFunction,
                FunctionSum,
                FunctionProduct,
    
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

include("hyperparameter.jl")
using MLKernels.HyperParameters: Bound, Interval, leftbounded, rightbounded, unbounded, checkbounds,
    Variable, fixed, Argument, HyperParameter

include("meta.jl")
include("functions.jl")
include("pairwise.jl")
#include("kernelfunction.jl")

end # MLKernels
