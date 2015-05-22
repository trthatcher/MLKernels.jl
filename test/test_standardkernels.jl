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
    SquaredExponentialKernel,
    GammaExponentialKernel,
    InverseQuadraticKernel,
    RationalQuadraticKernel,
    PowerKernel, 
    LogKernel,
    PeriodicKernel,
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
        (SquaredExponentialKernel, (1,), (2,)),
        (GammaExponentialKernel, (1, 0.5), (2, 0.3)),
        (InverseQuadraticKernel, (1,), (2,)),
        (RationalQuadraticKernel, (1, 1), (2, 2)),
        (PowerKernel, (1,), (0.5,)),
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

println("- Testing StandardKernel edge and special cases:")
for (kernelobject, case) in (
        # Edge Cases
        (LinearKernel, (0.0,)),
        (PolynomialKernel, (1.0, 0.0, 2.0)),
        (SigmoidKernel, (1.0, 0.0)),
        # Special Cases
        (LogKernel, (1,)),
        (PolynomialKernel, (1.0, 1.0, 2))
    )
    print("    - Testing ", kernelobject, case, " ... ")
    check_fields((kernelobject)(case...), case)
    println("Done")

end

println("- Testing StandardKernel error cases:")
for (kernelobject, error_case) in (
        (SquaredExponentialKernel, (-1,)),
        (GammaExponentialKernel, (-1, 0.5)),
        (GammaExponentialKernel, (1, 0)),
        (GammaExponentialKernel, (-1, 2)),
        (GammaExponentialKernel, (-1,)),
        (InverseQuadraticKernel, (-1,)),
        (RationalQuadraticKernel, (-1,)),
        (PowerKernel, (0,)),
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
        (SquaredExponentialKernel,      (1,),      exp(-1), true),
        (GammaExponentialKernel,        (1, 0.5),  exp(-1), true),
        (InverseQuadraticKernel,        (1,),      0.5,     true),
        (RationalQuadraticKernel,       (1, 1),    0.5,     true),
        (PowerKernel,                   (1,),      -1,      false),
        (LogKernel,                     (1,),      -log(2), false),
        (LinearKernel,                  (1,),      3,       true),
        (PolynomialKernel,              (1, 1, 2), 9,       true),
        (SigmoidKernel,                 (1, 1),    tanh(3), false),
        (MercerSigmoidKernel,           (0, 1),    tanh(1)*tanh(2), true))
    print("    - Testing ", kernelobject, " miscellaneous functions ... ")
    for T in (Float32, Float64)
        x, y = [one(T)], [convert(T,2)]
     
        κ = (kernelobject)(map(x -> convert(T, x), default_args)...)

        if kernelobject <: SquaredDistanceKernel
            u = MLKernels.sqdist(x, y)
            v = MLKernels.kappa(κ, u)
            @test_approx_eq v convert(T, default_value)
        end

        if kernelobject <: ScalarProductKernel
            u = dot(x, y)
            v = MLKernels.kappa(κ, u)
            @test_approx_eq v convert(T, default_value)
        end

        v = kernel(κ, x, y) # test on vectors
        @test_approx_eq v convert(T, default_value)

        v = kernel(κ, x[1], y[1]) # test on scalars
        @test_approx_eq v convert(T, default_value)

        @test isposdef(κ) == posdef
        
        for S in (Float32, Float64)

            @test convert(kernelobject{S}, κ) == (kernelobject)(map(x -> convert(S, x), default_args)...)

            if kernelobject <: SquaredDistanceKernel
                @test convert(SquaredDistanceKernel{S}, κ) == (kernelobject)(map(x -> convert(S, x), default_args)...)
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
        @test_approx_eq MLKernels.kappa_array!(κ, [one(T)])[1] convert(T, default_value)
    end
end

println("- Testing ARD kernels:")
print("    - Testing ARD{ScalarProductKernel} ... ")
for T in (Float32, Float64)
    x1, y1, d  = T[1, 3, 2], T[4, -2, 2], 3
    x2, y2, w2 = T[2, 3, 1], T[4, -2, 2], T[sqrt(0.5), 1.0, sqrt(2.0)]
    # weighted dot product = 2 in each case
    k1 = ARD(LinearKernel(convert(T, 1)), d)
    k2 = ARD(LinearKernel(convert(T, 1)), w2)
    for (k, x, y) in ((k1, x1, y1), (k2, x2, y2))
        @test_approx_eq kernel(k, x, y) convert(T, 3)
    end
end
println("Done")
print("    - Testing ARD{SquaredDistanceKernel} ... ")
for T in (Float32, Float64)
    x1, y1, d  = T[0, 1, 2], T[-1, 1, 2], 3
    x2, y2, w2 = T[0, 1, 2], T[0, 1.5, 3], T[1, sqrt(2), sqrt(0.5)]
    args = (1,)
    result = exp(-1)
    # weighted squared distance = 1 in each case
    k1 = ARD(SquaredExponentialKernel(map(x->convert(T,x), args)...), d)
    k2 = ARD(SquaredExponentialKernel(map(x->convert(T,x), args)...), w2)
    for (k, x, y) in ((k1, x1, y1), (k2, x2, y2))
        @test_approx_eq kernel(k, x, y) convert(T, result)
    end
end
println("Done")

