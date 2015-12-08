for kernel_obj in (additive_kernels..., composition_classes...)
    info("Testing ", kernel_obj)
    for T in FloatingPointTypes
    
        if T == Float64
            k = (kernel_obj)()
            @test eltype(k) == Float64
        end

        @test eltype(convert(kernel_obj{Float16},  (kernel_obj)())) == Float16
        @test eltype(convert(kernel_obj{Float32},  (kernel_obj)())) == Float32
        @test eltype(convert(kernel_obj{Float64},  (kernel_obj)())) == Float64
        @test eltype(convert(kernel_obj{BigFloat}, (kernel_obj)())) == BigFloat

        fields, default_args, test_args = get(all_args, kernel_obj, (Symbol[],T[],T[]))
        f = get(all_kernelfunctions, kernel_obj, "error")

        # Test Constructors
        for i = 1:length(fields)
            arg_values = (T[test_args[1:i]..., default_args[i+1:end]...]...)
            k = (kernel_obj)(arg_values...)
            @test eltype(k) == T
            for j = 1:length(fields)
                @test getfield(k, fields[j]) === arg_values[j]
            end
        end

        # Test Error Cases
        for test_args in get(all_errorcases, kernel_obj, ())
            arg_values = T[test_args...]
            @test_throws ErrorException (kernel_obj)(arg_values...)
        end

        # Test phi() function for all cases
        for test_args in get(all_phicases, kernel_obj, "error")
            arg_values = T[test_args...]
            k = convert(kernel_obj{T}, (kernel_obj)(arg_values...))
            for test_input in get(all_testinputs, kernel_obj, "error")
                z = T[test_input...]
                a = MOD.phi(k, z...)
                b = f(arg_values..., z...)
                if T == BigFloat
                    @test_approx_eq convert(Float64,a) convert(Float64,b)
                else
                    @test_approx_eq a b
                end
            end

        end


    end # End Floating Point loop

    k = (kernel_obj)()

    (atzero,atpos,atneg,ismer,isnd) = get(all_kernelproperties, kernel_obj, "error")
    @test MOD.attainszero(k) == atzero
    @test MOD.attainspositive(k) == atpos
    @test MOD.attainsnegative(k) == atneg
    @test ismercer(k) === ismer
    @test isnegdef(k) === isnd

    @test isa(MOD.description_string(k,true), AbstractString)
    @test isa(MOD.description_string(k,false), AbstractString)

end

info("Testing ", ARD)
for kernelobj in additive_kernels
    w = rand(3)
    for T in FloatingPointTypes
        k_base = convert(Kernel{T}, (kernelobj)())
        k = ARD(k_base, T[w...])

        @test eltype(k) == T
        @test eltype(convert(ARD{Float32}, k))  == Float32
        @test eltype(convert(ARD{Float64}, k))  == Float64
        @test eltype(convert(ARD{BigFloat}, k)) == BigFloat

        @test getfield(k, :k) == k_base 
        @test getfield(k, :w) == T[w...]
        
        @test MOD.attainszero(k) == MOD.attainszero(k_base)
        @test MOD.attainspositive(k) == MOD.attainspositive(k_base)
        @test MOD.attainsnegative(k) == MOD.attainsnegative(k_base)
        @test ismercer(k) === ismercer(k_base)
        @test isnegdef(k) === isnegdef(k_base)

        @test isa(MOD.description_string(k,true), AbstractString)
        @test isa(MOD.description_string(k,false), AbstractString)

    end  # End floating point loop
end  # End additive kernels loop

#=
info("Testing ", KernelSum)
for kernelobj1 in (SquaredDistanceKernel, RationalQuadraticKernel)
    for kernelobj2 in (ScalarProductKernel, PowerKernel)
        for T in FloatingPointTypes

            k1 = convert(Kernel{T}, (kernelobj1)())
            k2 = convert(Kernel{T}, (kernelobj2)())

            kvec = [k1, k2]

            if all(ismercer, kvec) || all(isnegdef, kvec)

                k = k1 + k2
                @test k.a == zero(T)
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

                @test eltype(convert(KernelSum{Float32}, k))  == Float32
                @test eltype(convert(KernelSum{Float64}, k))  == Float64
                @test eltype(convert(KernelSum{BigFloat}, k)) == BigFloat

                @test ismercer(k) == (ismercer(k1) && ismercer(k2))
                @test isnegdef(k) == (isnegdef(k1) && isnegdef(k2))
                @test attainszero(k) == (attainszero(k1) && attainszero(k2) && k.a == 0)
                @test ispositive(k) == (isnonnegative(k1) && isnonnegative(k2) && (ispositive(k1) || ispositive(k2)
                                                                                                  || k.a > 0)) 
                @test isnonnegative(k) == (isnonnegative(k1) && isnonnegative(k2))

                @test isa(MOD.description_string(k,true), AbstractString)
                @test isa(MOD.description_string(k,false), AbstractString)

                a = one(T)
                k = a + k1

                @test k.a == a
                @test k.k[1] == k1
                @test (k1 + a).k[1] == k.k[1]
                @test (k1 + a).a == k.a
                @test (k + a).a == 2a
                @test (a + k).a == 2a

                k = (k1 + a) + (k2 + a)
                @test k.a == 2a
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

                k = (k1 + a) + k2
                @test k.a == a
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

                k = k1 + (k2 + a)
                @test k.a == a
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

            else

                @test_throws ErrorException k1 + k2

            end
        end
    end
end
=#
#=
info("Testing ", KernelProduct)
for kernelobj1 in (SquaredDistanceKernel, RationalQuadraticKernel)
    for kernelobj2 in (ExponentialKernel, PolynomialKernel, LogKernel)
        for T in FloatingPointTypes

            k1 = convert(Kernel{T}, (kernelobj1)())
            k2 = convert(Kernel{T}, (kernelobj2)())

            kvec = [k1, k2]

            if all(ismercer, kvec)

                k = k1 * k2
                @test k.a == one(T)
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

                @test eltype(convert(KernelProduct{Float32},  k)) == Float32
                @test eltype(convert(KernelProduct{Float64},  k)) == Float64
                @test eltype(convert(KernelProduct{BigFloat}, k)) == BigFloat

                @test ismercer(k) == (ismercer(k1) && ismercer(k2))
                @test isnegdef(k) == (isnegdef(k1) && isnegdef(k2))
                @test attainszero(k) == (attainszero(k1) || attainszero(k2))
                @test ispositive(k) == (ispositive(k1) && ispositive(k2))
                @test isnonnegative(k) == (isnonnegative(k1) && isnonnegative(k2))

                @test isa(MOD.description_string(k,true), AbstractString)
                @test isa(MOD.description_string(k,false), AbstractString)

                a = convert(T,3)

                k = a * k1
                @test k.a == a
                @test k.k[1] == k1
                @test (k1 * a).k[1] == k.k[1]
                @test (k1 * a).a == k.a
                @test (k * a).a == a^2
                @test (a * k).a == a^2

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
