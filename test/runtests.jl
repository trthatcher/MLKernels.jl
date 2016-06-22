# Unit Tests
using Base.Test
using MLKernels

FloatingPointTypes = (Float32, Float64)
IntegerTypes = (Int32, UInt32, Int64, UInt64)
MOD = MLKernels

#=
module MLKTest

    using MLKernels

    import Base.convert

    immutable TestKernel{T<:AbstractFloat} <: StandardKernel{T}
        a::MLKernels.HyperParameter{T}
        TestKernel(a::Variable{T}) = new(HyperParameter(a, unbounded(T)))
    end
    TestKernel{T<:AbstractFloat}(a::Argument{T}=1.0) = TestKernel{T}(Variable(a))
    MLKernels.phi(κ::TestKernel, x::Real, y::Real) = x*y
    MLKernels.phi(κ::TestKernel, x::Vector, y::Vector) = product(x)*product(y)

    immutable PairwiseTestKernel{T<:AbstractFloat} <: PairwiseKernel{T}
        a::MLKernels.HyperParameter{T}
        PairwiseTestKernel(a::Variable{T}) = new(HyperParameter(a, unbounded(T)))
    end
    eval(MLKernels.generate_outer_constructor(PairwiseTestKernel, (1,)))
    eval(MLKernels.generate_conversions(PairwiseTestKernel))

    PairwiseTestKernel{T<:AbstractFloat}(a::Argument{T}=1.0) = PairwiseTestKernel{T}(Variable(a))
    MLKernels.pairwise(κ::PairwiseTestKernel, x::Real, y::Real) = x*y
    function MLKernels.unsafe_pairwise(κ::PairwiseTestKernel, x::AbstractArray, y::AbstractArray)
        prod(x)*prod(y)
    end

    immutable TestClass{T<:AbstractFloat} <: CompositionClass{T}
        a::MLKernels.HyperParameter{T}
        TestClass(a::Variable{T}) = new(HyperParameter(a, unbounded(T)))
    end
    TestClass{T<:AbstractFloat}(a::Argument{T}=1.0) = TestClass{T}(Variable(a))
end
=#

include("reference.jl")

include("hyperparameter.jl")
include("functions/pairwisefunction.jl")
include("functions/compositefunction.jl")
include("functions/pointwisefunction.jl")
#include("pairwisefunction.jl")
#include("kernelfunction.jl")
