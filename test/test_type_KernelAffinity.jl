info("Testing ", KernelAffinity)
for kernelobj in (additive_kernels..., composition_kernels...)
    for T in FloatingPointTypes

        k1 = convert(Kernel{T}, (kernelobj)())
        a = 2one(T)
        c = 3one(T)

        k = KernelAffinity(a, c, k1)
        @test k.a.value == a
        @test k.c.value == c

        @test eltype(convert(KernelAffinity{Float16},  k)) == Float16
        @test eltype(convert(KernelAffinity{Float32},  k)) == Float32
        @test eltype(convert(KernelAffinity{Float64},  k)) == Float64
        @test eltype(convert(KernelAffinity{BigFloat}, k)) == BigFloat

        @test MOD.attainszero(k) == MOD.attainszero(k.kappa)
        @test MOD.attainspositive(k) == MOD.attainspositive(k.kappa)
        @test MOD.attainsnegative(k) == MOD.attainsnegative(k.kappa)
        @test ismercer(k) === ismercer(k.kappa)
        @test isnegdef(k) === isnegdef(k.kappa)

        k = k1 + c
        @test k.a == one(T)
        @test k.c == c

        k = c + ((c + k1) + c)
        @test k.a == one(T)
        @test k.c == 3c

        k = k1 * a
        @test k.a == a
        @test k.c == zero(T)

        k = a * ((a * k1) * a)
        @test k.a == a^3
        @test k.c == zero(T)

        k = (a * k1) + c
        @test k.a == a
        @test k.c == c

        k = c + (k1 * a)
        @test k.a == a
        @test k.c == c

        k = a * (a * k1 + c)
        @test k.a == a^2
        @test k.c == a*c

        k = (a * k1 + c) * a
        @test k.a == a^2
        @test k.c == a*c

       #= k = a * convert(Kernel{T}, SquaredDistanceKernel()) + c
        @test k^(one(T)/2) == KernelComposition(PowerClass(a, c, one(T)/2), k.kappa)
        
        k = a * convert(Kernel{T}, ScalarProductKernel()) + c
        @test k^3     == KernelComposition(PolynomialClass(a, c, 3one(T)), k.kappa)
        @test exp(k)  == KernelComposition(ExponentiatedClass(a, c), k.kappa)
        @test tanh(k) == KernelComposition(SigmoidClass(a, c), k.kappa)=#

        # Test that output does not create error
        show(DevNull, k)

    end
end
