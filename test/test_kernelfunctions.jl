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
    @test_approx_eq kernelmatrix(k, X, false, true, true) K
    @test_approx_eq kernelmatrix(k, X, false, false, true) K
    print(" _Xt!")
    @test_approx_eq kernelmatrix(k, X', true, true, true) K
    @test_approx_eq kernelmatrix(k, X', true, false, true) K

    K = [kernel(ARD(k,w),x,y) for x in Set_x, y in Set_x]
    print(" w_X!")
    @test_approx_eq kernelmatrix(ARD(k,w), X, false, true, true) K
    @test_approx_eq kernelmatrix(ARD(k,w), X, false, false, true) K
    print(" w_Xt!")
    @test_approx_eq kernelmatrix(ARD(k,w), X', true, true, true) K
    @test_approx_eq kernelmatrix(ARD(k,w), X', true, false, true) K

    K = [kernel(k,x,y) for x in Set_x, y in Set_y]
    print(" _XY!")
    @test_approx_eq kernelmatrix(k, X, Y, false) K
    print(" _XtYt!")
    @test_approx_eq kernelmatrix(k, X', Y', true) K

    K = [kernel(ARD(k,w),x,y) for x in Set_x, y in Set_y]
    print(" w_XY!")
    @test_approx_eq kernelmatrix(ARD(k,w), X, Y, false) K
    print(" w_XtYt!")
    @test_approx_eq kernelmatrix(ARD(k,w), X', Y', true) K
    
    println(" ... Done")
end

println("Composite Kernel kernel() and kernelmatrix():")

for (kernelobject_comp, kernelobject_base) in (
                (ExponentialKernel, SquaredDistanceKernel),
                (RationalQuadraticKernel, SquaredDistanceKernel),
                #(MaternKernel, SquaredDistanceKernel),
                (PowerKernel, SquaredDistanceKernel),
                (LogKernel, SquaredDistanceKernel),
                (PolynomialKernel, ScalarProductKernel),
                (ExponentiatedKernel, ScalarProductKernel),
                (SigmoidKernel, ScalarProductKernel)
        )

    print(indent_block, kernelobject_comp)
    k_base = (kernelobject_base)()
    k_comp = (kernelobject_comp)(k_base)

    print(" Scalar")
    @test kernel(k_comp, x1[1], y1[1]) == MLKernels.phi(k_comp, MLKernels.pairwise(k_base, x1[1], y1[1]))
    
    print(" Vector")
    @test kernel(k_comp, x1, y1) == MLKernels.phi(k_comp, MLKernels.pairwise(k_base, x1, y1))

    print(" Matrix")
    K = [kernel(k_comp,x,y) for x in Set_x, y in Set_x]
    print(" _X!")
    @test_approx_eq kernelmatrix(k_comp, X, false, true, true) K
    @test_approx_eq kernelmatrix(k_comp, X, false, false, true) K
    print(" _Xt!")
    @test_approx_eq kernelmatrix(k_comp, X', true, true, true) K
    @test_approx_eq kernelmatrix(k_comp, X', true, false, true) K

    K = [kernel(k_comp,x,y) for x in Set_x, y in Set_y]
    print(" _XY!")
    @test_approx_eq kernelmatrix(k_comp, X, Y, false) K
    print(" _XtYt!")
    @test_approx_eq kernelmatrix(k_comp, X', Y', true) K

    println(" ... Done")

end
