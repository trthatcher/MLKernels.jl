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

println("-- Testing dot_rows --")
matrix_test_approx_eq(MLKernels.dot_rows([1.0 1 ; 0 1]), [2.0; 1])

println("-- Testing hadamard! --")
matrix_test_approx_eq(MLKernels.hadamard!([1.0; 1], [3.0; 2]), [3.0; 2])

X = reshape([1.0; 2], 2, 1)
Y = reshape([1.0; 1], 2, 1)

println("-- Testing gramian_matrix --")
matrix_test_approx_eq(gramian_matrix(X), [1.0 2; 2 4])
matrix_test_approx_eq(gramian_matrix(X, Y), [1.0 1; 2 2])

matrix_test_approx_eq(lagged_gramian_matrix(X), [0.0 1; 1 0.0])
matrix_test_approx_eq(lagged_gramian_matrix(X, Y), [0.0 0; 1 1])
