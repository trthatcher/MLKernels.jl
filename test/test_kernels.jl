using Base.Test

importall MLKernels

FloatingPointTypes = (Float32, Float64, BigFloat)

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

println("Additive Kernel Constructors:")

for kernelobj in additive_kernels

    info("Testing ", kernelobj)

    for T in FloatingPointTypes
    
        if T == Float64
            k = (kernelobj)()
            @test eltype(k) == Float64
        end

        # Test constructors
        fields, default_values, test_values = get(additive_kernelargs, kernelobj, (Symbol[],T[],T[]))
        for i = 1:length(fields)
            test_args = T[test_values[1:i]..., default_values[i+1:end]...]
            k = (kernelobj)(test_args...)
            @test eltype(k) == T
            for j = 1:length(fields)
                @test getfield(k, fields[j]) === test_args[j]
            end
        end

        for test_values in get(additive_errorcases, kernelobj, ())
            test_args = T[test_values...]
            @test_throws ErrorException (kernelobj)(test_args...)
        end

        #Test phi() function
        f = get(additive_kernelfunctions, kernelobj, "error")
        for test_values in get(additive_cases, kernelobj, "error")
            arg_values = T[test_values...]
            k = convert(Kernel{T},(kernelobj)(arg_values...))
            x = convert(T, x1[1])
            y = convert(T, y1[1])
            @test_approx_eq MLKernels.phi(k, x, y) f(arg_values..., x, y)
        end


    end

    @test ismercer((kernelobj)()) === get(additive_ismercer, kernelobj, "error")
    @test isnegdef((kernelobj)()) === get(additive_isnegdef, kernelobj, "error")

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
            fields, default_values, test_values = get(composite_kernelargs, kernelobj, (Symbol[],T[],T[]))
            for i = 1:length(fields)
                test_args = T[test_values[1:i]..., default_values[i+1:end]...]
                k = (kernelobj)(base_k, test_args...)
                @test eltype(k) == T
                @test getfield(k, :k) === base_k
                for j = 1:length(fields)
                    @test getfield(k, fields[j]) === test_args[j]
                end
            end

            for test_values in get(composite_errorcases, kernelobj, ())
                test_args = T[test_values...]
                @test_throws ErrorException (kernelobj)(base_k, test_args...)
            end

            # Test phi() function
            #=
            f = get(additive_kernelfunctions, kernelobj, "error")
            for test_values in get(additive_cases, kernelobj, "error")
                arg_values = T[test_values...]
                k = convert(Kernel{T},(kernelobj)(arg_values...))
                x = convert(T, x1[1])
                y = convert(T, y1[1])
                @test_approx_eq MLKernels.phi(k, x, y) f(arg_values..., x, y)
            end
            =#

        end

    end

    @test ismercer((kernelobj)()) === get(composite_ismercer, kernelobj, "error")
    @test isnegdef((kernelobj)()) === get(composite_isnegdef, kernelobj, "error")

end

T = Float64
