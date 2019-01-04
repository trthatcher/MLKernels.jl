for k in kernel_functions
    @testset "Testing $k" begin
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
                for j = 1:length(alt_args)
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

        ## Test Display
        #@test eval(Meta.parse(string(K))) == K
        #@test show(devnull, K) == nothing
    end
end

#@testset "Testing MLK.kappa" begin
#    for k in kernel_functions
#        k_tmp = get(kernel_functions_kappa, k, x->error(""))
#        for T in FloatingPointTypes
#            K = convert(Kernel{T}, (k)())
#            args = T[MLK.getvalue(getfield(K,theta)) for theta in fieldnames(typeof(K))]
#
#            for z in (zero(T), one(T), convert(T,2))
#                v = MLK.kappa(K, z)
#                v_tmp = (k_tmp)(z, args...)
#                @test isapprox(v, v_tmp)
#            end
#        end
#    end
#end
