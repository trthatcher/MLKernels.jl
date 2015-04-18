using Base.Test
importall MLKernels

function matrix_test_approx_eq(A::Array, B::Array)
    length(A) == length(B) || error("dimensions do not conform")
    for i = 1:length(A)
        @test_approx_eq A[i] B[i]
    end
end

println("-- Testing syml --")
matrix_test_approx_eq(MLKernels.syml([1.0 1 ; 0 1]), [1.0 1; 1 1])
matrix_test_approx_eq(MLKernels.symu([1.0 0 ; 1 1]), [1.0 1; 1 1])

println("-- Testing dot_rows --")
matrix_test_approx_eq(MLKernels.dot_rows([1.0 1 ; 0 1]), [2.0; 1])

println("-- Testing dot_columns --")
matrix_test_approx_eq(MLKernels.dot_columns([1.0 1 ; 0 1]), [1.0 2])

println("-- Testing hadamard! --")
matrix_test_approx_eq(MLKernels.hadamard!([1.0; 1], [3.0; 2]), [3.0; 2])

X = reshape([1.0; 2], 2, 1)
Y = reshape([1.0; 1], 2, 1)

println("-- Testing gramian_matrix --")
matrix_test_approx_eq(gramian_matrix(X), [1.0 2; 2 4])
matrix_test_approx_eq(gramian_matrix(X, Y), [1.0 1; 2 2])

matrix_test_approx_eq(lagged_gramian_matrix(X), [0.0 1; 1 0.0])
matrix_test_approx_eq(lagged_gramian_matrix(X, Y), [0.0 0; 1 1])

println("-- Testing center_kernel_matrix --")
X = [0.0 0; 2 2]
Z = X .- mean(X,1)

matrix_test_approx_eq(center_kernel_matrix(X*X'), Z*Z')

immutable TestKernel{T<:FloatingPoint} <: StandardKernel{T}
    a::T
end
kernel_function{T<:FloatingPoint}(::TestKernel{T}, x::Vector{T}, y::Vector{T}) = sum(x)*sum(y)

X = [0.0 0; 1 1]

println("-- Testing generic kernel_matrix --")
matrix_test_approx_eq(kernel_matrix(TestKernel(1.0), X), [0.0 0; 0 4])
matrix_test_approx_eq(kernel_matrix(TestKernel(1.0), X, X), [0.0 0; 0 4])

println("-- Testing generic kernel_matrix_scaled --")
matrix_test_approx_eq(MLKernels.kernel_matrix_scaled(2.0, TestKernel(1.0), X), [0.0 0; 0 8])
matrix_test_approx_eq(MLKernels.kernel_matrix_scaled(2.0, TestKernel(1.0), X, X), [0.0 0; 0 8])

println("-- Testing generic kernel_matrix_product --")
matrix_test_approx_eq(MLKernels.kernel_matrix_product(2.0, TestKernel(1.0), TestKernel(1.0), X), [0.0 0; 0 32])
matrix_test_approx_eq(MLKernels.kernel_matrix_product(2.0, TestKernel(1.0), TestKernel(1.0), X, X), [0.0 0; 0 32])

println("-- Testing generic kernel_matrix_sum --")
matrix_test_approx_eq(MLKernels.kernel_matrix_sum(2.0, TestKernel(1.0), 1.0, TestKernel(1.0), X), [0.0 0; 0 12])
matrix_test_approx_eq(MLKernels.kernel_matrix_sum(2.0, TestKernel(1.0), 1.0, TestKernel(1.0), X, X), [0.0 0; 0 12])
