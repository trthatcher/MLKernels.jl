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

    for kernel_comp in composition_classes
        for kernel_obj in get(composition_pairs, kernel_comp, "error")

            k_base = convert(Kernel{T}, kernel_obj())
            k_comp = convert(CompositionClass{T}, kernel_comp())
            k =  KernelComposition(k_comp, k_base)

            @test kernel(k, x[1], y[1]) == MLKernels.phi(k_comp, kernel(k_base, x[1], y[1]))
            @test (k)(x[1], y[1])       == MLKernels.phi(k_comp, kernel(k_base, x[1], y[1]))

            @test kernel(k, x, y) == MLKernels.phi(k_comp, kernel(k_base, x, y))
            @test (k)(x, y)       == MLKernels.phi(k_comp, kernel(k_base, x, y))
        end
    end

    for kernelobj1 in (RationalQuadraticKernel, GaussianKernel)
        for kernelobj2 in (PolynomialKernel, MaternKernel)
            k1 = convert(Kernel{T}, (kernelobj1)())
            k2 = convert(Kernel{T}, (kernelobj2)())

            k = one(T) + k1 + k2

            @test kernel(k, x[1], y[1]) == kernel(k1, x[1], y[1]) + kernel(k2, x[1], y[1]) + one(T)
            @test (k)(x[1], y[1])       == kernel(k1, x[1], y[1]) + kernel(k2, x[1], y[1]) + one(T)
            
            @test kernel(k, x, y) == kernel(k1, x, y) + kernel(k2, x, y) + one(T)
            @test (k)(x, y)       == kernel(k1, x, y) + kernel(k2, x, y) + one(T)

            k = convert(T,2) * k1 * k2

            @test kernel(k, x[1], y[1]) == kernel(k1, x[1], y[1]) * kernel(k2, x[1], y[1]) * 2one(T)
            @test (k)(x[1], y[1])       == kernel(k1, x[1], y[1]) * kernel(k2, x[1], y[1]) * 2one(T)

            @test kernel(k, x, y) == kernel(k1, x, y) * kernel(k2, x, y) * (2*one(T))
            @test (k)(x, y)       == kernel(k1, x, y) * kernel(k2, x, y) * (2*one(T))
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
        @test_approx_eq kernelmatrix(k, X, false, true, true)  K
        @test_approx_eq kernelmatrix(k, X, false, false, true) K
        @test_approx_eq kernelmatrix(k, X', true, true, true)  K
        @test_approx_eq kernelmatrix(k, X', true, false, true) K

        @test_approx_eq kernelmatrix(k, X) K

        K = [kernel(k,x,y) for x in Set_x, y in Set_y]
        @test_approx_eq kernelmatrix(k, X, Y, false)  K
        @test_approx_eq kernelmatrix(k, X', Y', true) K

        @test_approx_eq (k)(X, Y) K

    end

    for (kernelobj1, kernelobj2, kernelobj3) in (
            (RationalQuadraticKernel, PolynomialKernel, GaussianKernel),
            (ScalarProductKernel, MaternKernel, LaplacianKernel),
            (ChiSquaredKernel, SquaredDistanceKernel, SineSquaredKernel)
        )
        k1 = convert(Kernel{T},(kernelobj1)())
        k2 = convert(Kernel{T},(kernelobj2)())
        k3 = convert(Kernel{T},(kernelobj3)())
        k  = k1 + k2 + k3 + one(T)

        K = [kernel(k1,x,y) for x in Set_x, y in Set_x] +
            [kernel(k2,x,y) for x in Set_x, y in Set_x] +
            [kernel(k3,x,y) for x in Set_x, y in Set_x] .+ one(T)

        @test_approx_eq kernelmatrix(k, X, false, true, true)  K
        @test_approx_eq kernelmatrix(k, X, false, false, true) K
        @test_approx_eq kernelmatrix(k, X', true, true, true)  K
        @test_approx_eq kernelmatrix(k, X', true, false, true) K

        @test_approx_eq kernelmatrix(k, X) K

        K = [kernel(k1,x,y) for x in Set_x, y in Set_y] +
            [kernel(k2,x,y) for x in Set_x, y in Set_y] +
            [kernel(k3,x,y) for x in Set_x, y in Set_y] .+ one(T)

        @test_approx_eq kernelmatrix(k, X,  Y,  false)  K
        @test_approx_eq kernelmatrix(k, X', Y', true)   K

        @test_approx_eq kernelmatrix(k, X, Y) K

    end

    for (kernelobj1, kernelobj2, kernelobj3) in (
            (RationalQuadraticKernel, PolynomialKernel, GaussianKernel),
        )
        k1 = convert(Kernel{T},(kernelobj1)())
        k2 = convert(Kernel{T},(kernelobj2)())
        k3 = convert(Kernel{T},(kernelobj3)())
        k  = k1 * k2 * k3 * convert(T,2)

        K = [kernel(k1,x,y) for x in Set_x, y in Set_x] .*
            [kernel(k2,x,y) for x in Set_x, y in Set_x] .*
            [kernel(k3,x,y) for x in Set_x, y in Set_x] * convert(T,2)

        @test_approx_eq kernelmatrix(k, X, false, true, true)  K
        @test_approx_eq kernelmatrix(k, X, false, false, true) K
        @test_approx_eq kernelmatrix(k, X', true, true, true)  K
        @test_approx_eq kernelmatrix(k, X', true, false, true) K

        @test_approx_eq kernelmatrix(k, X) K

        K = [kernel(k1,x,y) for x in Set_x, y in Set_y] .*
            [kernel(k2,x,y) for x in Set_x, y in Set_y] .*
            [kernel(k3,x,y) for x in Set_x, y in Set_y] * convert(T,2)

        @test_approx_eq kernelmatrix(k, X,  Y,  false)  K
        @test_approx_eq kernelmatrix(k, X', Y', true)   K

        @test_approx_eq kernelmatrix(k, X, Y) K

    end

end

info("Testing ", centerkernelmatrix)
for T in (Float32, Float64)
    A = T[1 2 3;
          2 3 4;
          3 4 5]

    a = mean(A,1)

    @test_approx_eq centerkernelmatrix(A) ((A .- a) .- a') .+ mean(A)
end
