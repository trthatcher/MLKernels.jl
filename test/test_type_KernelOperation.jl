info("Testing ", KernelSum)
for kernelobj1 in (additive_kernels..., composition_kernels...)
    for kernelobj2 in (additive_kernels..., composition_kernels...)
        for T in FloatingPointTypes
            k1 = convert(Kernel{T}, (kernelobj1)())
            k2 = convert(Kernel{T}, (kernelobj2)())
            if ismercer(k1) && ismercer(k2) || isnegdef(k1) && isnegdef(k2)
                k = k1 + k2
                @test k.c.value == zero(T)
                #@test k.kappa1 == k1
                #@test k.kappa2 == k2

                @test eltype(convert(Kernel{Float32}, k))  == Float32
                @test eltype(convert(Kernel{Float64}, k))  == Float64
                @test eltype(convert(Kernel{BigFloat}, k)) == BigFloat

                @test ismercer(k) == (ismercer(k1) && ismercer(k2))
                @test isnegdef(k) == (isnegdef(k1) && isnegdef(k2))

                c = convert(T,2)

                k = (k1 + c) + (k2 + c)
                @test k.c.value == 2c
                #@test k.kappa1 == k1
                #@test k.kappa2 == k2

                k = (k1 + c) + k2
                @test k.c == c
                #@test k.kappa1 == k1
                #@test k.kappa2 == k2

                k = k1 + (k2 + c)
                @test k.c == c
                #@test k.kappa1 == k1
                #@test k.kappa2 == k2

            else
                @test_throws ErrorException k1 + k2
            end
        end
    end
end

#=
info("Testing ", KernelProduct)
for kernelobj1 in (SquaredDistanceKernel, RationalQuadraticKernel)
    for kernelobj2 in (ScalarProductKernel, ChiSquaredKernel)
        for T in FloatingPointTypes

            k1 = convert(Kernel{T}, (kernelobj1)())
            k2 = convert(Kernel{T}, (kernelobj2)())

            kvec = [k1, k2]

            if all(ismercer, kvec)

                k = k1 * k2
                @test k.a == one(T)
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

                @test eltype(convert(Kernel{Float32}, k))  == Float32
                @test eltype(convert(Kernel{Float64}, k))  == Float64
                @test eltype(convert(Kernel{BigFloat}, k)) == BigFloat

                @test ismercer(k) == (ismercer(k1) && ismercer(k2))

                @test MOD.attainszero(k) == any(MOD.attainszero, kvec)
                @test MOD.attainspositive(k) == any(MOD.attainspositive, kvec)
                @test MOD.attainsnegative(k) == any(MOD.attainsnegative, kvec)

                @test isa(MOD.description_string(k,true), AbstractString)
                @test isa(MOD.description_string(k,false), AbstractString)

                a = 2one(T)

                k = (k1 * a) * (k2 * a)
                @test k.a == a^2
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

                k = (k1 * a) * k2
                @test k.a == a
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

                k = k1 * (k2 * a)
                @test k.a == a
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

            else

                @test_throws ErrorException k1 * k2

            end
        end
    end
end
=#
