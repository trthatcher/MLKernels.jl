importall MLKernels

using Base.Test

x32 = [1.0f0]
y32 = [1.0f0]
x64 = [1.0]
y64 = [1.0]

testcase = 1

for (kernel, default_args, edgecase_args, error_args) in (
        (GaussianKernel, (1.0,), (), (-1.0,)),
        (LaplacianKernel, (1.0,), (), (-1.0,)),
        (RationalQuadraticKernel, (1.0,), (), (-1.0,)),
        (MultiQuadraticKernel, (1.0,), (), (-1.0,)),
        (InverseMultiQuadraticKernel, (1.0,), (), (-1.0,)),
        (PowerKernel, (2.0,), (), (-1.0,)),
        (LogKernel, (1.0,), (), (-1.0,)),
        (LinearKernel, (1.0,), (0.0,), (-1.0,)),
        (PolynomialKernel, (1.0, 1.0, 2.0), (1.0, 0.0, 2.0), (0.0, -1.0, 0.0)),
        (SigmoidKernel, (1.0, 1.0), (1.0, 0.0), (-1.0, -1.0)))
    κ = (kernel)()
    fields = fieldnames(κ)
    n = length(fields)
    println("Test case $(testcase): Construct ", kernel, " with arguments ()")
    testcase += 1
    for i = 1:n
        @test getfield(κ, fields[i]) == default_args[i]
    end
    for T in (Float32, Float64)
        base_args = map(T, default_args)
        for i = 1:n
            case_args = map(x -> 2*x, base_args[1:i])
            println("Test case $(testcase): Construct ", kernel, " with arguments ", case_args)
            testcase += 1
            κ = (kernel)(case_args...)
            for j = 1:n
                @test getfield(κ, fields[j]) == (j <= i ? case_args[j] : base_args[j])
            end
        end
        if length(edgecase_args) != 0
            case_args = map(T, edgecase_args)
            println("Test case $(testcase): Construct ", kernel, " with edge-case arguments ", case_args)
            testcase += 1
            @test (x->true)((kernel)(case_args...))
        end
        for i = 1:n
            case_args = i == 1 ? (T(error_args[i]),) : tuple(base_args[1:(i-1)]..., T(error_args[i]))
            println("Test case $(testcase): Construct ", kernel, " with invalid arguments ", case_args)
            testcase += 1
            for j = 1:n
                @test_throws ArgumentError (kernel)(case_args...)
            end
        end

    end
end
