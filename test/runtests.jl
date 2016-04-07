# Unit Tests
using Base.Test
using MLKernels

FloatingPointTypes = (Float32, Float64)
IntegerTypes = (Int32, UInt32, Int64, UInt64)
MOD = MLKernels


module MLKTest

    using MLKernels

    immutable TestKernel{T<:AbstractFloat} <: StandardKernel{T}
        a::MLKernels.HyperParameter{T}
        TestKernel(a::Variable{T}) = new(HyperParameter(a, unbounded(T)))
    end
    TestKernel{T<:AbstractFloat}(a::Argument{T}=1.0) = TestKernel{T}(Variable(a))
    MLKernels.phi(κ::TestKernel, x::Real, y::Real) = x*y
    MLKernels.phi(κ::TestKernel, x::Vector, y::Vector) = product(x)*product(y)

    immutable TestClass{T<:AbstractFloat} <: CompositionClass{T}
        a::MLKernels.HyperParameter{T}
        TestClass(a::Variable{T}) = new(HyperParameter(a, unbounded(T)))
    end
    TestClass{T<:AbstractFloat}(a::Argument{T}=1.0) = TestClass{T}(Variable(a))

end

include("reference.jl")

include("test_module_HyperParameter.jl")
include("test_type_PairwiseKernel.jl")
include("test_type_CompositionClass.jl")
include("test_type_KernelComposition.jl")
include("test_type_KernelAffinity.jl")
include("test_pairwise.jl")
#include("test_kernelfunctions.jl")
#include("test_kernelapproximation.jl")
