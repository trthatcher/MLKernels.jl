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

    println(indent_block, "Testing ", kernelobj, ":")

    for T in FloatingPointTypes

        print(indent_block^2, "Testing ",  T, " Constructor: ")        
        if T == Float64
            k = (kernelobj)()
            @test eltype(k) == Float64
        end

        fields, default_values, test_values = get(additive_kernelargs, kernelobj, (Symbol[],T[],T[]))
        for i = 1:length(fields)
            print(fields[i], " ")
            test_args = T[test_values[1:i]..., default_values[i+1:end]...]
            k = (kernelobj)(test_args...)
            @test eltype(k) == T
            for j = 1:length(fields)
                @test getfield(k, fields[j]) === test_args[j]
            end
        end

        println("... Done")

        print(indent_block^2, "Testing ",  T, " phi(): ")
        println("... Done")
    end

    print(indent_block^2, "Testing ismercer() ")
    println("... Done")

    print(indent_block^2, "Testing isnegdef() ")
    println("... Done")

end

T = Float64
