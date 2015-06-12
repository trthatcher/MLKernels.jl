using Base.Test
importall MLKernels

function matrix_test_approx_eq(A::Array, B::Array)
    length(A) == length(B) || error("dimensions do not conform")
    for i = 1:length(A)
        @test_approx_eq A[i] B[i]
    end
end

function test_kernelmatrix(K::Kernel, X::Matrix, reference::Matrix)
    matrix_test_approx_eq(MLKernels.kernelmatrix(K, X), reference)
    matrix_test_approx_eq(MLKernels.kernelmatrix(K, X, 'N', 'L'), reference)
    matrix_test_approx_eq(MLKernels.kernelmatrix(K, X', 'T'), reference)
    matrix_test_approx_eq(MLKernels.kernelmatrix(K, X', 'T', 'L'), reference)
    matrix_test_approx_eq(MLKernels.kernelmatrix(K, X, X), reference)
    matrix_test_approx_eq(MLKernels.kernelmatrix(K, X', X', 'T'), reference)
end

print("- Testing center_kernelmatrix ... ")
X = [0.0 0; 2 2]
Z = X .- mean(X,1)
matrix_test_approx_eq(center_kernelmatrix(X*X'), Z*Z')
println("Done")

module TestKernelModule
using MLKernels
import MLKernels.kernel
immutable TestKernel{T<:FloatingPoint} <: StandardKernel{T}
    a::T
end
kernel{T<:FloatingPoint}(::TestKernel{T}, x::Array{T}, y::Array{T}) = sum(x)*sum(y)
end
import TestKernelModule.TestKernel


X = [0.0 0; 1 1]

print("- Testing generic kernelmatrix ... ")
test_kernelmatrix(TestKernel(1.0), X, [0.0 0; 0 4])
println("Done")

print("- Testing generic kernelmatrix_scaled ... ")
test_kernelmatrix(2.0 * TestKernel(1.0), X, [0.0 0; 0 8])
println("Done")

print("- Testing generic kernelmatrix_product ... ")
test_kernelmatrix(2.0 * TestKernel(1.0) * TestKernel(1.0), X, [0.0 0; 0 32])
println("Done")

print("- Testing generic kernelmatrix_sum ... ")
test_kernelmatrix(2.0 * TestKernel(1.0) + 1.0 * TestKernel(1.0), X, [0.0 0; 0 12])
println("Done")


X = [1.0 0; 0 1]

K = PowerKernel(1.0)

print("- Testing optimized euclidean distance kernelmatrix ... ")
test_kernelmatrix(K, X, [0.0 -2; -2 0])
println("Done")

print("- Testing optimized euclidian distance kernelmatrix_scaled ... ")
test_kernelmatrix(2.0 * K, X, [0.0 -4; -4 0])
println("Done")

print("- Testing optimized euclidian distance kernelmatrix_product ... ")
test_kernelmatrix(2.0 * K * K, X, [0.0 8; 8 0])
println("Done")

print("- Testing optimized euclidian distance kernelmatrix_sum ... ")
test_kernelmatrix(2.0 * K + 1.0 * K, X, [0.0 -6; -6 0])
println("Done")

for (kernelobject, mercer) in (
        (ExponentialKernel, true),
        (RationalQuadraticKernel, true),
        (PowerKernel, false),
        (LogKernel, false),
        (PolynomialKernel, true),
    )
    print("- Testing ", kernelobject, " for", (mercer ? "" : " conditional"), " positive definity... ")
    for i = 1:10
        X = rand(10,5)
        c = 2*rand(10) .- 1
        c = mercer ? c : c .- mean(c)
        K = kernelmatrix((kernelobject)(), X)
        @test dot(c, K*c) >= 0
    end
    println("Done")
end


#=
X = [1.0 0; 0 1]

print("- Testing optimized separable kernel kernelmatrix ... ")
test_kernelmatrix(MercerSigmoidKernel(0.0, 1.0), X, [tanh(1.0)^2 0; 0 tanh(1.0)^2])
println("Done")

print("- Testing optimized separable kernel kernelmatrix_scaled ... ")
test_kernelmatrix(2.0 * MercerSigmoidKernel(0.0, 1.0), X, 2*[tanh(1.0)^2 0; 0 tanh(1.0)^2])
println("Done")
=#
