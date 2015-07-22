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

println("Additive Kernel pairwise():")
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
    @test MLKernels.pairwise(k, x1[1], y1[1]) == MLKernels.kappa(k, x1[1], y1[1])
    print(" Vector")
    @test MLKernels.pairwise(k, x1, y1) == MLKernels.kappa(k, x1[1], y1[1]) + MLKernels.kappa(k, x1[2], y1[2])
    print(" Matrix")
    P = [MLKernels.pairwise(k, x,y) for x in Set_x, y in Set_x]
    print(" _X!")
    matrix_test_approx_eq(MLKernels.syml!(MLKernels.pairwise(k, X,false,true)), P)
    matrix_test_approx_eq(MLKernels.symu!(MLKernels.pairwise(k, X,false,false)), P)
    print(" _Xt!")
    matrix_test_approx_eq(MLKernels.syml!(MLKernels.pairwise(k, X',true,true)), P)
    matrix_test_approx_eq(MLKernels.symu!(MLKernels.pairwise(k, X',true,false)), P)

    P = [MLKernels.pairwise(k, x,y) for x in Set_x, y in Set_y]
    print(" _XY!")
    matrix_test_approx_eq(MLKernels.pairwise(k, X, Y, false), P)
    print(" _XtYt!")
    matrix_test_approx_eq(MLKernels.pairwise(k, X', Y', true), P)

    println(" ... Done")
end


