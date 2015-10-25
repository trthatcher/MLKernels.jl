using Base.Test

importall MLKernels

macro test_approx_eq_type(value, reference, typ)
    x = gensym()
    quote
        $x = $value
        @test_approx_eq $x $reference
        @test isa($x, $typ)
    end
end

# Check each field for equality with args (assumed same order)
function check_fields(kernelobject::Kernel, field_values)
    fields = names(kernelobject)
    for i = 1:length(fields)
        @test getfield(kernelobject, fields[i]) == field_values[i]
    end
end

# Iterate through constructor cases - test only Float64 to ensure fields are populating as expected
function test_constructor_case{T<:Kernel}(kernelobject::Type{T}, default_args, test_args)
    n = length(names(kernelobject))
    check_fields((kernelobject)(), [default_args...])
    for i = 1:n
        case_args = [test_args[1:i]..., default_args[(i+1):end]...]
        κ = (kernelobject)(case_args[1:i]...)
        check_fields(κ, case_args)
    end
end



# Test Base Kernels

for kernelobj in additive_kernels

    info("Testing ", kernelobj)

    for T in FloatingPointTypes
    
        if T == Float64
            k = (kernelobj)()
            @test eltype(k) == Float64
        end

        # Test constructors
        fields, default_args, test_args = get(additive_kernelargs, kernelobj, (Symbol[],T[],T[]))
        for i = 1:length(fields)
            arg_values = T[test_args[1:i]..., default_args[i+1:end]...]
            k = (kernelobj)(arg_values...)
            @test eltype(k) == T
            for j = 1:length(fields)
                @test getfield(k, fields[j]) === arg_values[j]
            end
        end

        for test_args in get(additive_errorcases, kernelobj, ())
            arg_values = T[test_args...]
            @test_throws ErrorException (kernelobj)(arg_values...)
        end

        #Test phi() function
        f = get(additive_kernelfunctions, kernelobj, "error")
        for test_args in get(additive_cases, kernelobj, "error")
            arg_values = T[test_args...]
            k = convert(Kernel{T},(kernelobj)(arg_values...))
            for test_inputs in get(additive_testinputs, kernelobj, "error")
                x = convert(T, test_inputs[1])
                y = convert(T, test_inputs[2])
                a = MLKernels.phi(k, x, y)
                b = f(arg_values..., x, y)
                if T == BigFloat
                    @test_approx_eq convert(Float64,a) convert(Float64,b)
                else
                    @test_approx_eq a b
                end
            end
        end

    end

    k = (kernelobj)()

    @test ismercer(k) === get(additive_ismercer, kernelobj, "error")
    @test isnegdef(k) === get(additive_isnegdef, kernelobj, "error")


    (kr,az,pos,nn,np,neg) = get(additive_kernelranges, kernelobj, "error")
    @test MLKernels.kernelrange(k) == kr
    @test attainszero(k) == az
    @test ispositive(k) == pos
    @test isnonnegative(k) == nn
    @test isnonpositive(k) == np
    @test isnegative(k) == neg

end

for kernelobj in composite_kernels

    info("Testing ", kernelobj)

    for T in FloatingPointTypes
    
        if T == Float64
            k = (kernelobj)()
            @test eltype(k) == Float64
        end

        for base_kernelobj in get(composite_pairs, kernelobj, ())

            base_k = convert(Kernel{T}, (base_kernelobj)())

            # Test constructors
            fields, default_args, test_args = get(composite_kernelargs, kernelobj, (Symbol[],T[],T[]))
            for i = 1:length(fields)
                arg_values = T[test_args[1:i]..., default_args[i+1:end]...]
                k = (kernelobj)(base_k, arg_values...)
                @test eltype(k) == T
                @test getfield(k, :k) === base_k
                for j = 1:length(fields)
                    @test getfield(k, fields[j]) === arg_values[j]
                end
            end

            for test_args in get(composite_errorcases, kernelobj, ())
                arg_values = T[test_args...]
                @test_throws ErrorException (kernelobj)(base_k, arg_values...)
            end

            # Test phi() function
            f = get(composite_kernelfunctions, kernelobj, "error")
            for test_args in get(composite_cases, kernelobj, "error")
                arg_values = T[test_args...]
                k = convert(Kernel{T},(kernelobj)(base_k, arg_values...))
                for test_input in get(composite_testinputs, kernelobj, "error")
                    z = convert(T, test_input[1])
                    a = MLKernels.phi(k, z)
                    b = f(arg_values..., z)
                    if T == BigFloat
                        @test_approx_eq convert(Float64,a) convert(Float64,b)
                    else
                        @test_approx_eq a b
                    end
                end

            end

        end

    end

    @test ismercer((kernelobj)()) === get(composite_ismercer, kernelobj, "error")
    @test isnegdef((kernelobj)()) === get(composite_isnegdef, kernelobj, "error")

end

T = Float64
