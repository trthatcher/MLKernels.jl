#= Test Constructors =#

for k in kernel_functions
    @testset "Testing $k" begin
        def_args, alt_args = get(kernel_functions_arguments, k, ((), ()))

        # Test Constructors
        for T in FloatingPointTypes
            for i = 1:length(alt_args)
                K = (k)([typeof(θ) <: AbstractFloat ? convert(T,θ) : θ for θ in alt_args[1:i]]...)
                for j = 1:length(alt_args)
                    @test MOD.getvalue(getfield(K,j)) == (j <= i ? alt_args[j] : def_args[j])
                end
            end
        end

        # Test Conversions
        K = (k)()
        for psi in (Kernel, MercerKernel, NegativeDefiniteKernel)
            if k <: psi
                for T1 in FloatingPointTypes
                    T2 = T1 == Float64 ? Float32 : T1
                    @test eltype(convert(psi{T2},K)) == T2
                end
            end
        end

        # Test Properties
        @test MOD.ismercer(K) == (typeof(K) <: MOD.MercerKernel ? true : false)
        @test MOD.isnegdef(K) == (typeof(K) <: MOD.NegativeDefiniteKernel ? true : false)
        @test MODPF.isstationary(K) == MODPF.isstationary(MOD.pairwisefunction(K))
        @test MODPF.isisotropic(K) == MODPF.isisotropic(MOD.pairwisefunction(K))

        # Test Display
        @test eval(Meta.parse(string(K))) == K
        @test show(devnull, K) == nothing
    end
end

@testset "Testing MOD.kappa" begin
    for k in kernel_functions
        k_tmp = get(kernel_functions_kappa, k, x->error(""))
        for T in FloatingPointTypes
            K = convert(Kernel{T}, (k)())
            args = T[MOD.getvalue(getfield(K,theta)) for theta in fieldnames(typeof(K))]

            for z in (zero(T), one(T), convert(T,2))
                v = MOD.kappa(K, z)
                v_tmp = (k_tmp)(z, args...)
                @test isapprox(v, v_tmp)
            end
        end
    end
end

@testset "Testing MOD.gettheta" begin
    for k in kernel_functions
        K = (k)()
        K_theta = MOD.gettheta(K)
        @test K_theta == [MOD.gettheta(getfield(K,field)) for field in MOD.thetafieldnames(K)]
        @test MOD.checktheta(K, K_theta) == true
    end
end

@testset "Testing MOD.settheta!" begin
    for T in FloatingPointTypes
        alpha1 = one(T)
        alpha2 = convert(T,0.6)
        gamma1 = one(T)
        gamma2 = convert(T,0.4)
        K = MOD.GammaExponentialKernel(alpha1,gamma1)
        MOD.settheta!(K, [log(alpha2); log(gamma2)])
        @test getvalue(K.alpha) == alpha2
        @test getvalue(K.gamma) == gamma2

        @test_throws DimensionMismatch MOD.settheta!(K, [one(T)])
        @test_throws DimensionMismatch MOD.settheta!(K, [one(T); one(T); one(T)])
    end
end


@testset "Testing MOD.checktheta" begin
    for T in FloatingPointTypes
        K = MOD.GammaExponentialKernel(one(T),one(T))

        @test MOD.checktheta(K, [log(one(T)); log(one(T))]) == true
        @test MOD.checktheta(K, [log(one(T)); log(convert(T,2))]) == false

        @test_throws DimensionMismatch MOD.checktheta(K, [one(T)])
        @test_throws DimensionMismatch MOD.checktheta(K, [one(T); one(T); one(T)])
    end
end
