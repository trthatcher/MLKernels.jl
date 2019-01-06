function test_kernel_function(k)
    default_args, alt_args = get(kernel_functions_arguments, k, ((), ()))

    local n = length(default_args)
    K = (k)()

    @testset "Testing constructors" begin
        for j = 1:n
            @test getfield(K,j) == default_args[j]
        end
        @test eltype(K) == Float64

        for T in FloatingPointTypes, i = 1:n
            K = (k)([T(a) for a in alt_args[1:i]]...)
            for j = 1:n
                @test getfield(K,j) == (j <= i ? alt_args[j] : default_args[j])
            end
            @test eltype(K) == T
        end
    end

    @testset "Testing properties" begin
        @test MLK.ismercer(K) == isa(K, MLK.MercerKernel)
        @test MLK.isnegdef(K) == isa(K, MLK.NegativeDefiniteKernel)
        @test MLK.isstationary(K) == MLK.isstationary(MLK.basefunction(K))
        @test MLK.isisotropic(K) == MLK.isisotropic(MLK.basefunction(K))
    end

    @testset "Testing conversions" begin
        psi = k
        while psi != Any
            for T1 in FloatingPointTypes, T2 in FloatingPointTypes
                K1 = k{T1}()
                K2 = convert(psi{T2}, K1)
                @test eltype(K2) == T2
                @test isa(K2, k)
            end
            psi = supertype(psi)
        end
    end

    @testset "Testing kappa function" begin
        f = get(kernel_functions_kappa, k, x->error(""))
        args1, args2 = get(kernel_functions_arguments, k, ((), ()))
        for i = 0:length(args1)
            args = [j <= i ? args1[j] : args2[j] for j in eachindex(args1)]
            for T in FloatingPointTypes
                K = k{T}(args...)
                for z in [T(0), T(1), T(2)]
                    v1 = MLK.kappa(K, z)
                    v2 = (f)(z, args...)
                    @test isapprox(v1, v2)
                end
            end
        end
    end

    @testset "Testing show function" begin
        @test show(devnull, K) == nothing
    end
end

for k in kernel_functions
    @testset "Testing $k" begin
        test_kernel_function(k)
    end
end