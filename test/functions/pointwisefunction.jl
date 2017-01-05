info("Testing ", AffineKernel.name.name)
for f_obj in (pairwise_functions..., composite_functions...)
    for T in FloatingPointTypes

        f1 = convert(RealKernel{T}, (f_obj)())
        a = convert(T,2)
        c = convert(T,3)

        h = AffineKernel(a, c, f1)
        @test h.a.value == a
        @test h.c.value == c

        @test eltype(convert(AffineKernel{Float16},  h)) == Float16
        @test eltype(convert(AffineKernel{Float32},  h)) == Float32
        @test eltype(convert(AffineKernel{Float64},  h)) == Float64
        @test eltype(convert(AffineKernel{BigFloat}, h)) == BigFloat

        @test MOD.attainszero(h) == MOD.attainszero(h.f)
        @test MOD.attainspositive(h) == MOD.attainspositive(h.f)
        @test MOD.attainsnegative(h) == MOD.attainsnegative(h.f)
        @test ismercer(h) === ismercer(h.f)
        @test isnegdef(h) === isnegdef(h.f)

        # Test that output does not create error
        @test show(DevNull, h) == nothing

        h = f1 + c
        @test h.a == one(T)
        @test h.c == c
        @test h.f == f1

        h = c + ((c + f1) + c)
        @test h.a == one(T)
        @test h.c == 3c
        @test h.f == f1

        h = f1 * a
        @test h.a == a
        @test h.c == zero(T)
        @test h.f == f1

        h = a * ((a * f1) * a)
        @test h.a == a^3
        @test h.c == zero(T)
        @test h.f == f1

        h = (a * f1) + c
        @test h.a == a
        @test h.c == c
        @test h.f == f1

        h = c + (f1 * a)
        @test h.a == a
        @test h.c == c
        @test h.f == f1

        h = a * (a * f1 + c)
        @test h.a == a^2
        @test h.c == a*c
        @test h.f == f1

        h = (a * f1 + c) * a
        @test h.a == a^2
        @test h.c == a*c
        @test h.f == f1

        h = a * convert(RealKernel{T}, SquaredEuclidean()) + c
        @test h^(one(T)/2) == CompositeKernel(PowerClass(a, c, one(T)/2), h.f)
        
        h = a * convert(RealKernel{T}, ScalarProduct()) + c
        @test h^3     == CompositeKernel(PolynomialClass(a, c, 3), h.f)
        @test exp(h)  == CompositeKernel(ExponentiatedClass(a, c), h.f)
        @test tanh(h) == CompositeKernel(SigmoidClass(a, c), h.f)

    end
end


info("Testing ", KernelSum.name.name)
for f_obj1 in (pairwise_functions..., composite_functions...)
    for f_obj2 in (pairwise_functions..., composite_functions...)
        for T in FloatingPointTypes
            f1 = convert(RealKernel{T}, (f_obj1)())
            f2 = convert(RealKernel{T}, (f_obj2)())
            h = f1 + f2
            @test h.c.value == zero(T)
            @test h.f == f1
            @test h.g == f2

            @test eltype(convert(RealKernel{Float32}, h))  == Float32
            @test eltype(convert(RealKernel{Float64}, h))  == Float64
            @test eltype(convert(RealKernel{BigFloat}, h)) == BigFloat

            @test ismercer(h) == (ismercer(f1) && ismercer(f2))
            @test isnegdef(h) == (isnegdef(f1) && isnegdef(f2))

            # Test that output does not create error
            @test show(DevNull, h) == nothing

            c = convert(T,2)

            h = (f1 + c) + (f2 + c)
            @test h.c.value == 2c
            @test h.f == f1
            @test h.g == f2

            h = (f1 + c) + f2
            @test h.c == c
            @test h.f == f1
            @test h.g == f2

            h = f1 + (f2 + c)
            @test h.c == c
            @test h.f == f1
            @test h.g == f2

            h = (2*f1) + (2*f2)
            @test h.c == zero(T)
            @test h.f == 2*f1
            @test h.g == 2*f2

            h = f1 + (2*f2)
            @test h.c == zero(T)
            @test h.f == f1
            @test h.g == 2*f2

            h = (2*f1) + f2
            @test h.c == zero(T)
            @test h.f == 2*f1
            @test h.g == f2

            h = (f1 + f2) + 1
            @test h.c == one(T)
            @test h.f == f1
            @test h.g == f2
        end
    end
end

info("Testing ", KernelProduct.name.name)
for f_obj1 in (pairwise_functions..., composite_functions...)
    for f_obj2 in (pairwise_functions..., composite_functions...)
        for T in FloatingPointTypes
            f1 = convert(RealKernel{T}, (f_obj1)())
            f2 = convert(RealKernel{T}, (f_obj2)())
            h = f1 * f2
            @test h.a.value == one(T)
            @test h.f == f1
            @test h.g == f2

            @test eltype(convert(RealKernel{Float32}, h))  == Float32
            @test eltype(convert(RealKernel{Float64}, h))  == Float64
            @test eltype(convert(RealKernel{BigFloat}, h)) == BigFloat

            @test ismercer(h) == (ismercer(f1) && ismercer(f2))

            # Test that output does not create error
            @test show(DevNull, h) == nothing

            a = convert(T,2)

            h = (f1 * a) * (f2 * a)
            @test h.a.value == a^2
            @test h.f == f1
            @test h.g == f2

            h = (f1 * a) * f2
            @test h.a == a
            @test h.f == f1
            @test h.g == f2

            h = f1 * (f2 * a)
            @test h.a == a
            @test h.f == f1
            @test h.g == f2

            h = (f1 + 1) * (f2 + 1)
            @test h.a.value == one(T)
            @test h.f == (f1 + 1)
            @test h.g == (f2 + 1)

            h = f1 * (f2 + 1)
            @test h.a.value == one(T)
            @test h.f == f1
            @test h.g == (f2 + 1)

            h = (f1 + 1) * f2
            @test h.a.value == one(T)
            @test h.f == (f1 + 1)
            @test h.g == f2

            h = (f1 * f2) * 2
            @test h.a == convert(T,2)
            @test h.f == f1
            @test h.g == f2
        end
    end
end
