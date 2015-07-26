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
    @test MLKernels.pairwise(k, x1[1], y1[1]) == MLKernels.phi(k, x1[1], y1[1])
    @test MLKernels.pairwise(ARD(k,w[1:1]), x1[1], y1[1]) == w[1]^2*MLKernels.phi(k, x1[1], y1[1])
    print(" Vector")
    @test MLKernels.pairwise(k, x1, y1) == MLKernels.phi(k, x1[1], y1[1]) + MLKernels.phi(k, x1[2], y1[2])
    @test MLKernels.pairwise(ARD(k,w), x1, y1) == w[1]^2*MLKernels.phi(k, x1[1], y1[1]) + w[2]^2*MLKernels.phi(k, x1[2], y1[2])
    print(" Matrix")
    P = [MLKernels.pairwise(k,x,y) for x in Set_x, y in Set_x]
    print(" _X!")
    @test_approx_eq MLKernels.syml!(MLKernels.pairwise(k, X,false,true)) P
    @test_approx_eq MLKernels.symu!(MLKernels.pairwise(k, X,false,false)) P
    print(" _Xt!")
    @test_approx_eq MLKernels.syml!(MLKernels.pairwise(k, X',true,true)) P
    @test_approx_eq MLKernels.symu!(MLKernels.pairwise(k, X',true,false)) P

    P = [MLKernels.pairwise(k,x,y,w) for x in Set_x, y in Set_x]
    print(" w_X!")
    @test_approx_eq MLKernels.syml!(MLKernels.pairwise(ARD(k,w), X,false,true)) P
    @test_approx_eq MLKernels.symu!(MLKernels.pairwise(ARD(k,w), X,false,false)) P
    print(" w_Xt!")
    @test_approx_eq MLKernels.syml!(MLKernels.pairwise(ARD(k,w), X',true,true)) P
    @test_approx_eq MLKernels.symu!(MLKernels.pairwise(ARD(k,w), X',true,false)) P

    P = [MLKernels.pairwise(k,x,y) for x in Set_x, y in Set_y]
    print(" _XY!")
    @test_approx_eq MLKernels.pairwise(k, X, Y, false) P
    print(" _XtYt!")
    @test_approx_eq MLKernels.pairwise(k, X', Y', true) P

    P = [MLKernels.pairwise(k,x,y,w) for x in Set_x, y in Set_y]
    print(" w_XY!")
    @test_approx_eq MLKernels.pairwise(ARD(k,w), X, Y, false) P
    print(" w_XtYt!")
    @test_approx_eq MLKernels.pairwise(ARD(k,w), X', Y', true) P

    println(" ... Done")
end


