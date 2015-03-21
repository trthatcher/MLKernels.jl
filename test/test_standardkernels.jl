using Base.Test

importall MLKernels

function test_constructor(kernel::DataType, default_args, test_args)
    fields = names(kernel)
    n = length(fields)
    for i = 1:n
        case_args = test_args[1:i]
        println("Test: ", kernel, case_args)
        κ = (kernel)(case_args...)
        for j = 1:n
            if j <= i
                @test getfield(κ, fields[j]) == test_args[j]
            else
                @test getfield(κ, fields[j]) == default_args[j]
            end
        end
    end
end

function test_constructor_case(kernel::DataType, args)
    fields = names(kernel)
    n = length(fields)
    println("Test: ", kernel, args)
    κ = (kernel)(args...)
    for i = 1:n
        @test getfield(κ, fields[i]) == args[i]
    end
end

function test_default_constructor(kernel::DataType, default_args)
    println("Test: ", kernel, "()")
    κ = (kernel)()
    fields = names(κ)
    n = length(fields)
    for i = 1:n
        @test getfield(κ, fields[i]) == default_args[i]
    end
end

for (kernel, default_args, test_args) in (
        (GaussianKernel, (1,), (2,)),
        (LaplacianKernel, (1,), (2,)),
        (RationalQuadraticKernel, (1,), (2,)),
        (MultiQuadraticKernel, (1,), (2,)),
        (InverseMultiQuadraticKernel, (1,), (2,)),
        (PowerKernel, (2,), (2,)),
        (LogKernel, (1,), (2,)),
        (LinearKernel, (1,), (2,)),
        (PolynomialKernel, (1, 1, 2), (2, 2, 2)),
        (SigmoidKernel, (1, 1), (2, 2)))
    test_default_constructor(kernel, default_args)
    for T in (Float32, Float64)
        test_constructor(kernel, map(x -> convert(T, x), default_args), map(x -> convert(T, x), test_args))
    end
end

println("Testing Standard Kernel Constructors:")
for (kernel, default_args, test_args) in (
        (GaussianKernel, (1,), (2,)),
        (LaplacianKernel, (1,), (2,)),
        (RationalQuadraticKernel, (1,), (2,)),
        (MultiQuadraticKernel, (1,), (2,)),
        (InverseMultiQuadraticKernel, (1,), (2,)),
        (PowerKernel, (2,), (2,)),
        (LogKernel, (1,), (2,)),
        (LinearKernel, (1,), (2,)),
        (PolynomialKernel, (1, 1, 2), (2, 2, 2)),
        (SigmoidKernel, (1, 1), (2, 2)))
    test_default_constructor(kernel, default_args)
    for T in (Float32, Float64)
        test_constructor(kernel, map(x -> convert(T, x), default_args), map(x -> convert(T, x), test_args))
    end
end

println("Testing Standard Kernel Edge Cases:")
for (kernel, edge_case_list) in (
        (LinearKernel, ((0,),)),
        (PolynomialKernel, ((1, 0, 2),)),
        (SigmoidKernel, ((1, 0),)))
    for T in (Float32, Float64)
        for edge_case in edge_case_list
            test_case = map(x -> convert(T, x), edge_case)
            test_constructor_case(kernel, test_case)
        end
    end
end

println("Testing Standard Kernel Special Cases:")
for (kernel, special_case) in (
        (PowerKernel, (2,)),
        (LogKernel, (1,)),
        (PolynomialKernel, (1.0, 1.0, 2)))
    test_constructor_case(kernel, special_case)
end

println("Testing Standard Kernel Error Cases:")
for (kernel, error_case_list) in (
        (GaussianKernel, ((-1,),)),
        (LaplacianKernel, ((-1,),)),
        (RationalQuadraticKernel, ((-1,),)),
        (MultiQuadraticKernel, ((-1,),)),
        (InverseMultiQuadraticKernel, ((-1.0,),)),
        (PowerKernel, ((-1,),)),
        (LogKernel, ((-1,),)),
        (LinearKernel, ((-1.0,),)),
        (PolynomialKernel, ((-1, 1, 2), (1, -1, 2), (1, 1, 0))),
        (SigmoidKernel, ((-1, 1), (1, -1))))
    for T in (Float32, Float64)
        for error_case in error_case_list
            test_case = map(x -> convert(T, x), error_case)
            println("Test:", kernel, test_case)
            @test_throws ArgumentError (kernel)(test_case...)
        end
    end
end

println("Testing Functions")
for (kernel, default_args, default_value, posdef) in (
        (GaussianKernel,                (1,), exp(-1),  true),
        (LaplacianKernel,               (1,), exp(-1),  true),
        (RationalQuadraticKernel,       (1,), 0.5,      true),
        (MultiQuadraticKernel,          (1,), sqrt(2),  false),
        (InverseMultiQuadraticKernel,   (1,), 1/sqrt(2),false),
        (PowerKernel,                   (2,), -1,       false),
        (LogKernel,                     (1,), -log(2),  false),
        (LinearKernel,                  (1,),      3,       true),
        (PolynomialKernel,              (1, 1, 2), 9,       true),
        (SigmoidKernel,                 (1, 1),    tanh(3), false))
    for T in (Float32, Float64)
        x, y = [one(T)], [convert(T,2)]
        
        if kernel <: EuclideanDistanceKernel
            u = MLKernels.euclidean_distance(x, y)
        end

        if kernel <: ScalarProductKernel
            u = MLKernels.scalar_product(x, y)
        end

        κ = (kernel)(map(x -> convert(T, x), default_args)...)
        show(κ)

        println("Test: kernelize_scalar(", κ, ",", u, ") == ", convert(T, default_value))
        v = MLKernels.kernelize_scalar(κ, u)
        @test_approx_eq v convert(T, default_value)

        println("Test: ", κ, (x, y), " == ", convert(T, default_value))
        v = kernel_function(κ, x, y)
        @test_approx_eq v convert(T, default_value)

        println("Test: isposdef_kernel(", κ, ") == ", posdef)
        @test isposdef_kernel(κ) == posdef

        println("Test: convert(",kernel{Float32}, ", ", κ, ")")
        @test convert(kernel{Float32}, κ) == (kernel)(map(x -> convert(Float32, x), default_args)...)

        println("Test: convert(",kernel{Float64}, ", ", κ, ")")
        @test convert(kernel{Float64}, κ) == (kernel)(map(x -> convert(Float64, x), default_args)...)

        if kernel <: EuclideanDistanceKernel

            println("Test: convert(",EuclideanDistanceKernel{Float32}, ", ", κ, ")")
            @test convert(EuclideanDistanceKernel{Float32}, κ) == (kernel)(map(x -> convert(Float32, x), default_args)...)

            println("Test: convert(",EuclideanDistanceKernel{Float64}, ", ", κ, ")")
            @test convert(EuclideanDistanceKernel{Float64}, κ) == (kernel)(map(x -> convert(Float64, x), default_args)...)

        end
        
        if kernel <: ScalarProductKernel

            println("Test: convert(",ScalarProductKernel{Float32}, ", ", κ, ")")
            @test convert(ScalarProductKernel{Float32}, κ) == (kernel)(map(x -> convert(Float32, x), default_args)...)

            println("Test: convert(",ScalarProductKernel{Float64}, ", ", κ, ")")
            @test convert(ScalarProductKernel{Float64}, κ) == (kernel)(map(x -> convert(Float64, x), default_args)...)   

        end

        println("Test: convert(",StandardKernel{Float32}, ", ", κ, ")")
        @test convert(StandardKernel{Float32}, κ) == (kernel)(map(x -> convert(Float32, x), default_args)...)

        println("Test: convert(",StandardKernel{Float64}, ", ", κ, ")")
        @test convert(StandardKernel{Float64}, κ) == (kernel)(map(x -> convert(Float64, x), default_args)...)

        println("Test: convert(",SimpleKernel{Float32}, ", ", κ, ")")
        @test convert(SimpleKernel{Float32}, κ) == (kernel)(map(x -> convert(Float32, x), default_args)...)

        println("Test: convert(",SimpleKernel{Float64}, ", ", κ, ")")
        @test convert(SimpleKernel{Float64}, κ) == (kernel)(map(x -> convert(Float64, x), default_args)...)

        println("Test: convert(",Kernel{Float32}, ", ", κ, ")")
        @test convert(Kernel{Float32}, κ) == (kernel)(map(x -> convert(Float32, x), default_args)...)

        println("Test: convert(",Kernel{Float64}, ", ", κ, ")")
        @test convert(Kernel{Float64}, κ) == (kernel)(map(x -> convert(Float64, x), default_args)...)
    end
    @test description((kernel)()) == Nothing()
end


