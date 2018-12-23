function test_base_function(f)
    F = (f)()

    @testset "Testing properties" begin
        test_properties = get(base_functions_properties, f, (false,false))
        @test MLK.isstationary(F) == test_properties[1]
        @test MLK.isisotropic(F)  == test_properties[2]
    end

    # Test that initial value is set correctly compared to reference
    @testset "Testing $(MLK.base_initiate)" begin
        for T in FloatingPointTypes
            s = MLK.base_initiate(F, T)
            s_tmp = convert(T, get(base_functions_initiate, f, 0))

            @test typeof(s) == T
            @test s == s_tmp
        end
    end

    # Test that aggregation works correctly
    @testset "Testing $(MLK.base_aggregate)" begin
        for T in FloatingPointTypes
            f_tmp = get(base_functions_aggregate, f, (s,x,y) -> convert(T, NaN))
            for x in (T(0), T(1), T(2)), y in (T(0), T(1), T(2))
                s = MLK.base_aggregate(F, T(1), x, y)
                s_tmp = f_tmp(T(1), x, y)

                @test typeof(s) == T
                @test s == s_tmp
            end
        end
    end

    # Test that return works correctly
    @testset "Testing $(MLK.base_return)" begin
        for T in FloatingPointTypes
            f_tmp = get(base_functions_return, f, s -> s)
            for x in (T(0), T(1), T(2))
                s = MLK.base_return(F, x)
                s_tmp = f_tmp(x)

                @test typeof(s) == T
                @test s == s_tmp
            end
        end
    end
end

for f in base_functions
    @testset "Testing $f" begin
        test_base_function(f)
    end
end