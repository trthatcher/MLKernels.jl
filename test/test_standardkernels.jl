using Base.Test

importall MLKernels

# Check each field for equality with args (assumed same order)
function check_fields(kernelobject::StandardKernel, args)
    fields = names(kernelobject)
    for i = 1:length(fields)
        @test getfield(kernelobject, fields[i]) == args[i]
    end
end

# Iterate through the test_args and test optional arguments
function test_constructor(kernelobject::DataType, default_args, test_args)
    fields = names(kernelobject)
    n = length(fields)
    for i = 1:n
        case_args = test_args[1:i]
        κ = (kernelobject)(case_args...)
        check_fields(κ, tuple(case_args..., default_args[i+1:n]...))
    end
end

println("Testing standard kernel output:")
for kernelobject in (
    GaussianKernel,
    LaplacianKernel,
    RationalQuadraticKernel,
    MultiQuadraticKernel,
    InverseMultiQuadraticKernel,
    PowerKernel,
    LogKernel,
    LinearKernel,
    PolynomialKernel,
    SigmoidKernel,
    MercerSigmoidKernel
)
    print(STDOUT, "    - Testing ")
    show(STDOUT, (kernelobject)())
    println(" ... Done")
end

println("- Testing StandardKernel constructors:")
for (kernelobject, default_args, test_args) in (
        (GaussianKernel, (1,), (2,)),
        (LaplacianKernel, (1,), (2,)),
        (RationalQuadraticKernel, (1,), (2,)),
        (MultiQuadraticKernel, (1,), (2,)),
        (InverseMultiQuadraticKernel, (1,), (2,)),
        (PowerKernel, (2,), (2,)),
        (LogKernel, (1,), (2,)),
        (LinearKernel, (1,), (2,)),
        (PolynomialKernel, (1, 1, 2), (2, 2, 4)),
        (SigmoidKernel, (1, 1), (2, 2)),
        (MercerSigmoidKernel, (0, 1), (-1, 2))
    )
    print("    - Testing ", kernelobject, " ... ")
    check_fields((kernelobject)(), default_args)
    for T in (Float32, Float64)
        case_defaults = map(x -> convert(T, x), default_args)
        case_tests = map(x -> convert(T, x), test_args)
        test_constructor(kernelobject, case_defaults, case_tests)
    end
    println("Done")
end

println("- Testing StandardKernel aliases:")
for (kernelobject, kernelalias, test_args) in (
        (GaussianKernel, SquaredExponentialKernel, (2,)),
        (LaplacianKernel, ExponentialKernel, (2,))
    )
    print("    - Testing ", kernelalias, " -> ", kernelobject, " ... ")
    for T in (Float32, Float64)
        case_tests = map(x -> convert(T, x), test_args)
        @test kernelobject() === kernelalias()
        @test kernelobject(case_tests...) === kernelalias(case_tests...)
    end
    println("Done")
end

println("- Testing StandardKernel edge and special cases:")
for (kernelobject, case) in (
        # Edge Cases
        (LinearKernel, (0.0,)),
        (PolynomialKernel, (1.0, 0.0, 2.0)),
        (SigmoidKernel, (1.0, 0.0)),
        # Special Cases
        (PowerKernel, (2,)),
        (LogKernel, (1,)),
        (PolynomialKernel, (1.0, 1.0, 2))
    )
    print("    - Testing ", kernelobject, case, " ... ")
    check_fields((kernelobject)(case...), case)
    println("Done")

end

println("- Testing StandardKernel error cases:")
for (kernelobject, error_case) in (
        (GaussianKernel, (-1,)),
        (LaplacianKernel, (-1,)),
        (RationalQuadraticKernel, (-1,)),
        (MultiQuadraticKernel, (-1,)),
        (InverseMultiQuadraticKernel, (-1.0,)),
        (PowerKernel, (-1,)),
        (LogKernel, (-1,)),
        (LinearKernel, (-1.0,)),
        (PolynomialKernel, (-1, 1, 2)), 
        (PolynomialKernel, (1, -1, 2)), 
        (PolynomialKernel, (1, 1, 0)),
        (SigmoidKernel, (-1, 1)),
        (SigmoidKernel, (1, -1)),
        (MercerSigmoidKernel, (0, 0)), 
        (MercerSigmoidKernel, (0, -1))
    )
    print("    - Testing ", kernelobject, error_case, " ... ")
    for T in (Float32, Float64)
        test_case = map(x -> convert(T, x), error_case)
        @test_throws ArgumentError (kernelobject)(test_case...)
    end
    println("Done")
end

println("- Testing miscellaneous functions:")
for (kernelobject, default_args, default_value, posdef) in (
        (GaussianKernel,                (1,), exp(-0.5),  true),
        (LaplacianKernel,               (1,), exp(-1),  true),
        (RationalQuadraticKernel,       (1,), 0.5,      true),
        (MultiQuadraticKernel,          (1,), sqrt(2),  false),
        (InverseMultiQuadraticKernel,   (1,), 1/sqrt(2),false),
        (PowerKernel,                   (2,), -1,       false),
        (LogKernel,                     (1,), -log(2),  false),
        (LinearKernel,                  (1,),      3,       true),
        (PolynomialKernel,              (1, 1, 2), 9,       true),
        (SigmoidKernel,                 (1, 1),    tanh(3), false),
        (MercerSigmoidKernel,           (0, 1),    tanh(1)*tanh(2), true))
    print("    - Testing ", kernelobject, " miscellaneous functions ... ")
    for T in (Float32, Float64)
        x, y = [one(T)], [convert(T,2)]
     
        κ = (kernelobject)(map(x -> convert(T, x), default_args)...)

        if kernelobject <: EuclideanDistanceKernel
            u = MLKernels.euclidean_distance(x, y)
            v = MLKernels.kernelize_scalar(κ, u)
            @test_approx_eq v convert(T, default_value)
        end

        if kernelobject <: ScalarProductKernel
            u = MLKernels.scalar_product(x, y)
            v = MLKernels.kernelize_scalar(κ, u)
            @test_approx_eq v convert(T, default_value)
        end

        v = kernel(κ, x, y) # test on vectors
        @test_approx_eq v convert(T, default_value)

        v = kernel(κ, x[1], y[1]) # test on scalars
        @test_approx_eq v convert(T, default_value)

        @test isposdef(κ) == posdef
        
        for S in (Float32, Float64)

            @test convert(kernelobject{S}, κ) == (kernelobject)(map(x -> convert(S, x), default_args)...)

            if kernelobject <: EuclideanDistanceKernel
                @test convert(EuclideanDistanceKernel{S}, κ) == (kernelobject)(map(x -> convert(S, x), default_args)...)
            end
        
            if kernelobject <: ScalarProductKernel
                @test convert(ScalarProductKernel{S}, κ) == (kernelobject)(map(x -> convert(S, x), default_args)...) 
            end

            if kernelobject <: SeparableKernel
                @test convert(SeparableKernel{S}, κ) == (kernelobject)(map(x -> convert(S, x), default_args)...) 
            end

            @test convert(StandardKernel{S}, κ) == (kernelobject)(map(x -> convert(S, x), default_args)...)
            @test convert(SimpleKernel{S}, κ) == (kernelobject)(map(x -> convert(S, x), default_args)...)
            @test convert(Kernel{S}, κ) == (kernelobject)(map(x -> convert(S, x), default_args)...)

        end

    end

    @test typeof(MLKernels.description_string_long((kernelobject)())) <: String

    println("Done")
end

for (kernelobject, default_args, default_value) in (
        (MercerSigmoidKernel, (0,1), tanh(1)),)
    for T in (Float32, Float64)
        κ = (kernelobject)(map(x -> convert(T, x), default_args)...)
        @test_approx_eq MLKernels.kernelize_array!(κ, [one(T)])[1] convert(T, default_value)
    end
end
