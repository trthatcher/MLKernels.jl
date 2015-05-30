println("- Testing StandardKernel show():")
for kernelobject in (
        ExponentialKernel,
        RationalQuadraticKernel,
        PowerKernel,
        LogKernel,
        PolynomialKernel,
        SigmoidKernel
    )
    print(STDOUT, "    - Testing ")
    show(STDOUT, (kernelobject)())
    println(" ... Done")
end

println("- Testing StandardKernel constructors:")
for (kernelobject, default_args, test_args) in (
        (ExponentialKernel, [1, 1], [2, 0.5]),
        (RationalQuadraticKernel, [1, 1, 1], [2, 2, 0.5]),
        (PowerKernel, [1], [0.5]),
        (LogKernel, [1,1], [2,0.5]),
        (PolynomialKernel, [1,1,2], [2,2,3]),
        (SigmoidKernel, [1,1], [2,2])
    )
    print("    - Testing ", kernelobject, " ... ")
    test_constructor_case(kernelobject, default_args, test_args)
    println("Done")
end

println("- Testing ARD constructors:")
for (kernelobject, test_args) in (
        (ExponentialKernel, [2, 0.5]),
        (RationalQuadraticKernel, [2, 2, 0.5]),
        (PowerKernel, [0.5]),
        (LogKernel, [2,0.5]),
        (PolynomialKernel, [2,2,3]),
        (SigmoidKernel, [2,2])
    )
    print("    - Testing ARD ", kernelobject, " ... ")
    for T in (Float32, Float64)
        w = [convert(T,2)]
        case_args = T[test_args...]
        K = ARD((kernelobject)(case_args...),w)
        @test K.weights == w
        check_fields(K.k, case_args)
        d = 3
        K = ARD((kernelobject)(case_args...),d)
        @test K.weights == ones(T,d)
        check_fields(K.k, case_args)
    end
    println("Done")
end


println("- Testing StandardKernel error cases:")
for (kernelobject, error_cases) in (
        (ExponentialKernel, ([0], [0, 1], [1, 0], [1, 2])),
        (RationalQuadraticKernel, ([0], [1, 0], [1, 1, 0], [1,1,1.01])),
        (PowerKernel, ([0],[1.0001])),
        (LogKernel, ([0],[1,0], [1,1.0001])),
        (PolynomialKernel, ([0,1,2], [1,-0.0001,3], [1,1,0], [1,1,1.5])),
        (SigmoidKernel, ([0,1], [1,-0.00001]))
    )
    print("    - Testing ", kernelobject, " error cases ... ")
    for error_case in error_cases
        print(" ", error_case)
        test_error_case(kernelobject, error_case)
    end
    println(" ... Done")
end

println("- Testing ismercer() property:")
for (kernelobject, mercer) in (
        (ExponentialKernel, true),
        (RationalQuadraticKernel, true),
        (PowerKernel, false),
        (LogKernel, false),
        (PolynomialKernel, true),
        (SigmoidKernel, false)
    )
    print("    - Testing ", kernelobject, "... ")
    @test ismercer((kernelobject)()) == mercer
    println("Done")
end

println("- Testing ScalarProductKernel kernel() function:")
for (kernelobject, test_args, test_function) in (
        (PolynomialKernel, [1,1,2], (z,a,c,d) -> (a * z + c )^d),
        (SigmoidKernel, [1,1], (z,a,c) -> tanh(a*z + c))
    )
    for is_ARD in (false, true)
        for T in (Float32, Float64)
            case_args = T[test_args...]
            x, y, w = T[1], T[2], T[2]
            K = is_ARD ? ARD((kernelobject)(case_args...),w) : (kernelobject)(case_args...)
            xy = is_ARD ? sum(x .* y .* w.^2) : sum(x .* y)
            test_value = test_function(xy, case_args...)
            print("    - Testing ", K, "... ")
            kernel_value = is_ARD ? MLKernels.kappa(K.k, dot(w .*x, w.*y)) : MLKernels.kappa(K, dot(x,y))
            @test_approx_eq kernel_value test_value
            @test isa(kernel_value,T)
            kernel_value = kernel(K,x,y)
            @test_approx_eq kernel_value test_value
            @test isa(kernel_value,T)
            kernel_value = kernel(K,x[1],y[1])
            @test_approx_eq kernel_value test_value
            @test isa(kernel_value,T)
            println("Done")
        end
    end
end

println("- Testing SquaredDistanceKernel kernel() function:")
for (kernelobject, test_args, test_function) in (
        (ExponentialKernel, [2,0.5], (z,a,t) -> exp(-a * z^t)),
        (RationalQuadraticKernel, [2,2,0.5], (z,a,b,t) -> (1 + a*z^t)^(-b)),
        (PowerKernel, [0.5], (z,t) -> -z^t),
        (LogKernel, [2,0.5], (z,a,t) -> -log(a*z^t + 1))
    )
    for is_ARD in (false,true)
        for T in (Float32, Float64)
            case_args = T[test_args...]
            x, y, w = T[1], T[2], T[2]
            K = is_ARD ? ARD((kernelobject)(case_args...), w) : (kernelobject)(case_args...)
            lag = is_ARD ? w .* (x - y) : x - y
            test_value = test_function(dot(lag,lag), case_args...)
            print("    - Testing ", K, "... ")
            kernel_value = MLKernels.kappa(is_ARD ? K.k : K,dot(lag,lag))
            @test_approx_eq kernel_value test_value
            @test isa(kernel_value,T)
            kernel_value = kernel(K,x,y)
            @test_approx_eq kernel_value test_value
            @test isa(kernel_value,T)
            kernel_value = kernel(K,x[1],y[1])
            @test_approx_eq kernel_value test_value
            @test isa(kernel_value,T)
            println("Done")
        end
    end
end
