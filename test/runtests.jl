# Unit Tests
using Base.Test
using MLKernels

FloatingPointTypes = (Float32, Float64)
IntegerTypes = (Int32, UInt32, Int64, UInt64)
MOD = MLKernels

test_print(f::RealFunction) = string(typeof(f).name)
test_print(h::AffineFunction) = string(typeof(h).name, "(", typeof(h.f).name.name, ")")
test_print(h::CompositeFunction) = string(typeof(h).name.name, "(", typeof(h.g).name.name, ",",
                                          typeof(h.f).name.name, ")")
test_print(h::FunctionSum) = string(typeof(h).name.name, "(", typeof(h.g).name.name, ",",
                                    typeof(h.f).name.name, ")")
test_print(h::FunctionProduct) = string(typeof(h).name.name, "(", typeof(h.g).name.name, ",",
                                        typeof(h.f).name.name, ")")
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
include("pairwise.jl")
