
info("Testing ", kernel)
for T in FloatingPointTypes

    x = T[1; 2]
    y = T[1; 1]
    w = T[2; 1]

    for kernelobj in additive_kernels
        k = convert(Kernel{T}, (kernelobj)())

        @test kernel(k, x[1], y[1]) == MLKernels.pairwise(k, x[1], y[1])
        @test (k)(x[1], y[1])       == MLKernels.pairwise(k, x[1], y[1])

        @test kernel(ARD(k,w[1:1]), x[1], y[1]) == MLKernels.pairwise(ARD(k,w[1:1]), x[1], y[1])
        @test (ARD(k,w[1:1]))(x[1], y[1])       == MLKernels.pairwise(ARD(k,w[1:1]), x[1], y[1])

        @test kernel(k, x, y) == MLKernels.pairwise(k, x, y)
        @test (k)(x, y)       == MLKernels.pairwise(k, x, y)

        @test kernel(ARD(k,w), x, y) == MLKernels.pairwise(ARD(k,w), x, y)
        @test (ARD(k,w))(x, y)       == MLKernels.pairwise(ARD(k,w), x, y)
    end

    for kernelobj in composite_kernels
        for base_kernelobj in get(composite_pairs, kernelobj, "error")

            k_base = convert(Kernel{T}, (base_kernelobj)())
            k = (kernelobj)(k_base)

            @test kernel(k, x[1], y[1]) == MLKernels.phi(k, MLKernels.pairwise(k_base, x[1], y[1]))
            @test (k)(x[1], y[1])       == MLKernels.phi(k, MLKernels.pairwise(k_base, x[1], y[1]))

            @test kernel(k, x, y) == MLKernels.phi(k, MLKernels.pairwise(k_base, x, y))
            @test (k)(x, y)       == MLKernels.phi(k, MLKernels.pairwise(k_base, x, y))
        end
    end

    for kernelobj1 in (RationalQuadraticKernel, ExponentialKernel)
        for kernelobj2 in (PolynomialKernel, MaternKernel)
            k1 = convert(Kernel{T}, (kernelobj1)())
            k2 = convert(Kernel{T}, (kernelobj2)())

            k = one(T) + k1 + k2

            @test kernel(k, x[1], y[1]) == (MLKernels.phi(k1, MLKernels.pairwise(k1.k, x[1], y[1])) +
                                            MLKernels.phi(k2, MLKernels.pairwise(k2.k, x[1], y[1])) + one(T))
            @test (k)(x[1], y[1])       == (MLKernels.phi(k1, MLKernels.pairwise(k1.k, x[1], y[1])) +
                                            MLKernels.phi(k2, MLKernels.pairwise(k2.k, x[1], y[1])) + one(T))

            @test kernel(k, x, y) == (MLKernels.phi(k1, MLKernels.pairwise(k1.k, x, y)) +
                                      MLKernels.phi(k2, MLKernels.pairwise(k2.k, x, y)) + one(T))
            @test (k)(x, y)       == (MLKernels.phi(k1, MLKernels.pairwise(k1.k, x, y)) +
                                      MLKernels.phi(k2, MLKernels.pairwise(k2.k, x, y)) + one(T))

            k = convert(T,2) * k1 * k2

            @test kernel(k, x[1], y[1]) == (MLKernels.phi(k1, MLKernels.pairwise(k1.k, x[1], y[1])) *
                                            MLKernels.phi(k2, MLKernels.pairwise(k2.k, x[1], y[1])) * convert(T,2))
            @test (k)(x[1], y[1])       == (MLKernels.phi(k1, MLKernels.pairwise(k1.k, x[1], y[1])) *
                                            MLKernels.phi(k2, MLKernels.pairwise(k2.k, x[1], y[1])) * convert(T,2))

            @test kernel(k, x, y) == (MLKernels.phi(k1, MLKernels.pairwise(k1.k, x, y)) *
                                      MLKernels.phi(k2, MLKernels.pairwise(k2.k, x, y)) * convert(T,2))
            @test (k)(x, y)       == (MLKernels.phi(k1, MLKernels.pairwise(k1.k, x, y)) *
                                      MLKernels.phi(k2, MLKernels.pairwise(k2.k, x, y)) * convert(T,2))
        end
    end
end


info("Testing ", kernelmatrix)
for T in FloatingPointTypes

    x1 = T[1; 2]
    x2 = T[2; 0]
    x3 = T[3; 2]
    X =  T[x1'; x2'; x3']

    y1 = T[1; 1]
    y2 = T[1; 1]
    Y = T[y1'; y2']

    w = T[2; 1]

    Set_x = (x1,x2,x3)
    Set_y = (y1,y2)

    for kernelobj in (RationalQuadraticKernel, PolynomialKernel, SquaredDistanceKernel)

        k = convert(Kernel{T},(kernelobj)())

        K = [kernel(k,x,y) for x in Set_x, y in Set_x]
        @test_approx_eq kernelmatrix(k, X, false, true, true) K
        @test_approx_eq kernelmatrix(k, X, false, false, true) K
        @test_approx_eq kernelmatrix(k, X', true, true, true) K
        @test_approx_eq kernelmatrix(k, X', true, false, true) K

        @test_approx_eq kernelmatrix(k, X) K

        K = [kernel(k,x,y) for x in Set_x, y in Set_y]
        @test_approx_eq kernelmatrix(k, X, Y, false) K
        @test_approx_eq kernelmatrix(k, X', Y', true) K

        @test_approx_eq (k)(X, Y) K

    end
end

#=
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
=#

T = Float64
