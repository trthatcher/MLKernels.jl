T = Float64

x1 = T[1; 2]
x2 = T[2; 0]
x3 = T[3; 2]
X = [x1'; x2'; x3']

y1 = T[1; 1]
y2 = T[1; 1]
Y = [y1'; y2']

w = T[2; 1]

Set_x = (x1,x2,x3)
Set_y = (y1,y2)

println("Additive Kernel kernel() and kernelmatrix():")
for kernelobject in (
        SquaredDistanceKernel,
        SineSquaredKernel,
        ChiSquaredKernel,
        ScalarProductKernel,
        MercerSigmoidKernel
    )
    print(indent_block, kernelobject)
    k = (kernelobject)()

    print(" Scalar")
    @test kernel(k, x1[1], y1[1]) == MLKernels.pairwise(k, x1[1], y1[1])
    @test kernel(ARD(k,w[1:1]), x1[1], y1[1]) == MLKernels.pairwise(ARD(k,w[1:1]), x1[1], y1[1])

    print(" Vector")
    @test kernel(k, x1, y1) == MLKernels.pairwise(k, x1, y1)
    @test kernel(ARD(k,w), x1, y1) == MLKernels.pairwise(ARD(k,w), x1, y1)

    print(" Matrix")
    K = [kernel(k,x,y) for x in Set_x, y in Set_x]
    print(" _X!")
    matrix_test_approx_eq(kernelmatrix(k, X, 'N', 'U', true), K)
    matrix_test_approx_eq(kernelmatrix(k, X, 'N', 'L', true), K)
    print(" _Xt!")
    matrix_test_approx_eq(kernelmatrix(k, X', 'T', 'U', true), K)
    matrix_test_approx_eq(kernelmatrix(k, X', 'T', 'L', true), K)

    K = [kernel(ARD(k,w),x,y) for x in Set_x, y in Set_x]
    print(" w_X!")
    matrix_test_approx_eq(kernelmatrix(ARD(k,w), X, 'N', 'U', true), K)
    matrix_test_approx_eq(kernelmatrix(ARD(k,w), X, 'N', 'L', true), K)
    print(" w_Xt!")
    matrix_test_approx_eq(kernelmatrix(ARD(k,w), X', 'T', 'U', true), K)
    matrix_test_approx_eq(kernelmatrix(ARD(k,w), X', 'T', 'L', true), K)

    K = [kernel(k,x,y) for x in Set_x, y in Set_y]
    print(" _XY!")
    matrix_test_approx_eq(kernelmatrix(k, X, Y, 'N'), K)
    print(" _XtYt!")
    matrix_test_approx_eq(kernelmatrix(k, X', Y', 'T'), K)

    K = [kernel(ARD(k,w),x,y) for x in Set_x, y in Set_y]
    print(" w_XY!")
    matrix_test_approx_eq(kernelmatrix(ARD(k,w), X, Y, 'N'), K)
    print(" w_XtYt!")
    matrix_test_approx_eq(kernelmatrix(ARD(k,w), X', Y', 'T'), K)
    
    println(" ... Done")
end

