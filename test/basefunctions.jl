for f in base_functions
    @testset "Testing $f" begin
        F = (f)()
        for T in FloatingPointTypes
            s = MLK.base_initiate(F, T)
            s_tmp = convert(T, get(base_functions_initiate, f, 0))

            @test typeof(s) == T
            @test s == s_tmp

            f_tmp = get(base_functions_aggregate, f, (s,x,y) -> convert(T, NaN))
            for x in (zero(T), one(T), convert(T,2)), y in (zero(T), one(T), convert(T,2))
                s = MLK.base_aggregate(F, one(T), x, y)
                s_tmp = f_tmp(one(T), x, y)

                @test typeof(s) == T
                @test s == s_tmp
            end

            f_tmp = get(base_functions_return, f, s -> s)
            for x in (zero(T), one(T), convert(T,2))
                s = MLK.base_return(F, x)
                s_tmp = f_tmp(x)

                @test typeof(s) == T
                @test s == s_tmp
            end

            test_properties = get(base_functions_properties, f, (false,false))
            @test MLK.isstationary(F) == test_properties[1]
            @test MLK.isisotropic(F)  == test_properties[2]
        end
    end
end
