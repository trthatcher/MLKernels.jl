using Base.Test

importall MLKernels

const FloatingPointTypes = (Float32, Float64, BigFloat)

macro test_approx_eq_type(value, reference, typ)
    x = gensym()
    quote
        $x = $value
        @test_approx_eq $x $reference
        @test isa($x, $typ)
    end
end

# Check each field for equality with args (assumed same order)
function check_fields(kernelobject::Kernel, field_values)
    fields = names(kernelobject)
    for i = 1:length(fields)
        @test getfield(kernelobject, fields[i]) == field_values[i]
    end
end

# Iterate through constructor cases - test only Float64 to ensure fields are populating as expected
function test_constructor_case{T<:Kernel}(kernelobject::Type{T}, default_args, test_args)
    n = length(names(kernelobject))
    check_fields((kernelobject)(), [default_args...])
    for i = 1:n
        case_args = [test_args[1:i]..., default_args[(i+1):end]...]
        κ = (kernelobject)(case_args[1:i]...)
        check_fields(κ, case_args)
    end
end



# Test Base Kernels

T = Float64

println("Standard Kernel Constructors:")
for (kernelobject, fields, default_fields, test_fields) in (
        (SquaredDistanceKernel, [:t], T[1], T[0.5]),
        (SineSquaredKernel, [:t], T[1], T[0.5]),
        (ChiSquaredKernel, [:t], T[1], T[0.5]),
        (ScalarProductKernel,[], [], []),
        (MercerSigmoidKernel, [:d, :b], T[0, 1], T[1, 2]),
        (ExponentialKernel, [:k, :alpha, :gamma], [SquaredDistanceKernel(), T[1,1]...], [SineSquaredKernel(), T[2,0.5]...]),
        (RationalQuadraticKernel, [:k, :alpha, :beta, :gamma], [SquaredDistanceKernel(), T[1,1,1]...], [SineSquaredKernel(), T[2,2,0.5]...]),
        (MaternKernel, [:k, :nu, :theta], [SquaredDistanceKernel(), T[1,1]...], [SineSquaredKernel(), T[2,2]...]),
        (PowerKernel, [:k, :gamma], [SquaredDistanceKernel(), T[1]...], [SineSquaredKernel(), T[0.5]...]),
        (LogKernel, [:k, :alpha, :gamma], [SquaredDistanceKernel(), T[1,1]...], [SineSquaredKernel(), T[2,0.5]...]),
        (PolynomialKernel, [:k, :alpha, :c, :d], [ScalarProductKernel(), T[1,1,2]...], [MercerSigmoidKernel(), T[2,0.5,3]...]),
        (ExponentiatedKernel, [:k, :alpha], [ScalarProductKernel(), T[1]...], [MercerSigmoidKernel(), T[2]...]),
        (SigmoidKernel, [:k, :alpha, :c], [ScalarProductKernel(), T[1,1]...], [MercerSigmoidKernel(), T[2,2]...])
    )
    print(indent_block, "Testing ", kernelobject, " [] ")
    (kernelobject)()
    for i = 1:length(fields)
        print(fields[i], " ")
        test_args = [test_fields[1:i]..., default_fields[i+1:end]...]
        k = (kernelobject)(test_args...)
        for j = 1:length(fields)
            @test getfield(k, fields[j]) === test_args[j]
        end
    end
    println("... Done")
end

println("Standard Kernel ismercer() and isnegdef():")

println("Base Kernel phi():")

println("Composite Kernel phi():")
