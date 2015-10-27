using Base.Test

importall MLKernels

# Test Base Kernels

for kernelobj in (additive_kernels..., composite_kernels...)
    info("Testing ", kernelobj)
    for T in FloatingPointTypes
    
        if T == Float64
            k = (kernelobj)()
            @test eltype(k) == Float64
        end

        @test eltype(convert(kernelobj{T}, (kernelobj)())) == T

        fields, default_args, test_args = get(all_kernelargs, kernelobj, (Symbol[],T[],T[]))

        f = get(all_kernelfunctions, kernelobj, "error")

        for base_kernelobj in get(composite_pairs, kernelobj, (kernelobj,))
            base_k = convert(Kernel{T}, (base_kernelobj)())
            is_composed = kernelobj in composite_kernels

            # Test Constructors
            for i = 1:length(fields)
                arg_values = (T[test_args[1:i]..., default_args[i+1:end]...]...)
                k = is_composed ? (kernelobj)(base_k, arg_values...) : (kernelobj)(arg_values...)
                @test eltype(k) == T
                if is_composed
                    @test getfield(k, :k) === base_k
                end
                for j = 1:length(fields)
                    @test getfield(k, fields[j]) === arg_values[j]
                end
            end

            # Test Error Cases
            for test_args in get(all_errorcases, kernelobj, ())
                arg_values = T[test_args...]
                if is_composed
                    @test_throws ErrorException (kernelobj)(base_k, arg_values...) 
                else
                    @test_throws ErrorException (kernelobj)(arg_values...)
                end
            end

            # Test phi() function for all cases
            for test_args in get(all_phicases, kernelobj, "error")
                arg_values = T[test_args...]
                k = convert(Kernel{T}, is_composed ? (kernelobj)(base_k, arg_values...) : (kernelobj)(arg_values...))
                for test_input in get(all_testinputs, kernelobj, "error")
                    z = T[test_input...]
                    a = MLKernels.phi(k, z...)
                    b = f(arg_values..., z...)
                    if T == BigFloat
                        @test_approx_eq convert(Float64,a) convert(Float64,b)
                    else
                        @test_approx_eq a b
                    end
                end

            end

        end # End Base Kernel loop
    end # End Floating Point loop

    k = (kernelobj)()

    (kr,az,pos,nn,np,neg,ismer,isnd) = get(all_kernelproperties, kernelobj, "error")
    @test MLKernels.kernelrange(k) == kr
    @test attainszero(k) == az
    @test ispositive(k) == pos
    @test isnonnegative(k) == nn
    @test isnonpositive(k) == np
    @test isnegative(k) == neg
    @test ismercer(k) === ismer
    @test isnegdef(k) === isnd

end

info("Testing ", ARD)
for kernelobj in additive_kernels
    w = rand(3)
    for T in FloatingPointTypes
        k_base = convert(Kernel{T}, (kernelobj)())
        k = ARD(k_base, T[w...])

        @test eltype(k) == T
        @test getfield(k, :k) == k_base 
        @test getfield(k, :w) == T[w...]
        @test MLKernels.kernelrange(k) == MLKernels.kernelrange(k_base)
        @test attainszero(k) == attainszero(k_base)
        @test ispositive(k) == ispositive(k_base)
        @test isnonnegative(k) == isnonnegative(k_base)
        @test isnonpositive(k) == isnonpositive(k_base)
        @test isnegative(k) == isnegative(k_base)
        @test ismercer(k) === ismercer(k_base)
        @test isnegdef(k) === isnegdef(k_base)

    end  # End floating point loop
end  # End additive kernels loop


T = Float64
