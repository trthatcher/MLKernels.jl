
for class_obj in composition_classes
    info("Testing ", class_obj)

    for T in FloatingPointTypes

        default_floats, default_others = all_default_args[class_obj]
        default_args = (T[default_floats...]..., default_others...)
        fields = fieldnames(class_obj)
        k = (class_obj)(default_args...)

        @test eltype(k) == T

        for i in eachindex(fields)
            @test getfield(k, fields[i]).value == default_args[i]
        end

        for z in (zero(T),one(T))
            f = all_phifunctions[class_obj]
            @test_approx_eq MOD.phi(k, z) f(default_args..., z)
        end
    end
end

for kernel_obj in additive_kernels
    info("Testing ", kernel_obj)

    for T in FloatingPointTypes

        default_floats, default_others = all_default_args[kernel_obj]
        default_args = (T[default_floats...]..., default_others...)
        fields = fieldnames(kernel_obj)
        k = length(fields) == 0 ? (kernel_obj){T}() : (kernel_obj)(default_args...)

        @test eltype(k) == T

        for i in eachindex(fields)
            @test getfield(k, fields[i]).value == default_args[i]
        end

        for (x,y) in ((zero(T),zero(T)), (zero(T),one(T)), (one(T),zero(T)), (one(T),one(T)))
            f = all_phifunctions[kernel_obj]
            @test_approx_eq MOD.phi(k, x, y) f(default_args..., x, y)
        end
    end
end


#for (x,y) in ((zero(T),zero(T)), (zero(T),one(T)), (one(T),zero(T)), (one(T),one(T)))



#=
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

info("Testing ", KernelComposition)
for comp_obj in composition_classes
    for kernel_obj in get(composition_pairs, comp_obj, "Error")
        for T in FloatingPointTypes
            k_comp = convert(CompositionClass{T}, (comp_obj)())
            k_base = convert(Kernel{T}, kernel_obj())
            k = KernelComposition(k_comp, k_base)

            @test k == k_comp ∘ k_base

            @test eltype(k) == T
            @test eltype(convert(Kernel{Float32}, k))  == Float32
            @test eltype(convert(Kernel{Float64}, k))  == Float64
            @test eltype(convert(Kernel{BigFloat}, k)) == BigFloat
            
            @test MOD.attainszero(k) == MOD.attainszero(k_comp)
            @test MOD.attainspositive(k) == MOD.attainspositive(k_comp)
            @test MOD.attainsnegative(k) == MOD.attainsnegative(k_comp)
            @test ismercer(k) === ismercer(k_comp)
            @test isnegdef(k) === isnegdef(k_comp)

            @test isa(MOD.description_string(k,true), AbstractString)
            @test isa(MOD.description_string(k,false), AbstractString)

        end
    end
end

for T in FloatingPointTypes

    k = convert(Kernel{T}, ScalarProductKernel())
    @test k^3     == KernelComposition(PolynomialClass(one(T), zero(T), 3one(T)), k)
    @test exp(k)  == KernelComposition(ExponentiatedClass(one(T), zero(T)), k)
    @test tanh(k) == KernelComposition(SigmoidClass(one(T), zero(T)), k)

    k = convert(Kernel{T}, SquaredDistanceKernel())
    @test k^(one(T)/2) == KernelComposition(PowerClass(one(T), zero(T), one(T)/2), k)

end

info("Testing ", KernelAffinity)
for kernelobj in (additive_kernels..., composition_kernels...)
    for T in FloatingPointTypes

        k1 = convert(Kernel{T}, (kernelobj)())
        a = 2one(T)
        c = 3one(T)

        k = KernelAffinity(a, c, k1)
        @test k.a == a
        @test k.c == c

        @test eltype(convert(KernelAffinity{Float16},  k)) == Float16
        @test eltype(convert(KernelAffinity{Float32},  k)) == Float32
        @test eltype(convert(KernelAffinity{Float64},  k)) == Float64
        @test eltype(convert(KernelAffinity{BigFloat}, k)) == BigFloat

        @test MOD.attainszero(k) == MOD.attainszero(k.k)
        @test MOD.attainspositive(k) == MOD.attainspositive(k.k)
        @test MOD.attainsnegative(k) == MOD.attainsnegative(k.k)
        @test ismercer(k) === ismercer(k.k)
        @test isnegdef(k) === isnegdef(k.k)

        @test isa(MOD.description_string(k,true), AbstractString)
        @test isa(MOD.description_string(k,false), AbstractString)

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

        k = a * convert(Kernel{T}, SquaredDistanceKernel()) + c
        @test k^(one(T)/2) == KernelComposition(PowerClass(a, c, one(T)/2), k.k)
        
        k = a * convert(Kernel{T}, ScalarProductKernel()) + c
        @test k^3     == KernelComposition(PolynomialClass(a, c, 3one(T)), k.k)
        @test exp(k)  == KernelComposition(ExponentiatedClass(a, c), k.k)
        @test tanh(k) == KernelComposition(SigmoidClass(a, c), k.k)

    end
end


info("Testing ", KernelSum)
for kernelobj1 in (SquaredDistanceKernel, RationalQuadraticKernel)
    for kernelobj2 in (ScalarProductKernel, ChiSquaredKernel)
        for T in FloatingPointTypes

            k1 = convert(Kernel{T}, (kernelobj1)())
            k2 = convert(Kernel{T}, (kernelobj2)())

            kvec = [k1, k2]

            if all(ismercer, kvec) || all(isnegdef, kvec)

                k = k1 + k2
                @test k.c == zero(T)
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

                @test eltype(convert(Kernel{Float32}, k))  == Float32
                @test eltype(convert(Kernel{Float64}, k))  == Float64
                @test eltype(convert(Kernel{BigFloat}, k)) == BigFloat

                @test ismercer(k) == (ismercer(k1) && ismercer(k2))
                @test isnegdef(k) == (isnegdef(k1) && isnegdef(k2))

                @test MOD.attainszero(k) == (all(MOD.attainszero, kvec) && k.c == 0) || (
                                         any(MOD.attainspositive, kvec) && any(MOD.attainsnegative, kvec))
                @test MOD.attainspositive(k) == any(MOD.attainspositive, kvec)
                @test MOD.attainsnegative(k) == any(MOD.attainsnegative, kvec)

                @test isa(MOD.description_string(k,true), AbstractString)
                @test isa(MOD.description_string(k,false), AbstractString)

                c = 2one(T)

                k = (k1 + c) + (k2 + c)
                @test k.c == 2c
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

                k = (k1 + c) + k2
                @test k.c == c
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

                k = k1 + (k2 + c)
                @test k.c == c
                @test all(k.k .== kvec) || all(k.k .== reverse(kvec))

            else

                @test_throws ErrorException k1 + k2

            end
        end
    end
end


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
