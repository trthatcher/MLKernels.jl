for f in pairwise_functions
    @testset "Testing $f" begin
        F = (f)()
        for T in FloatingPointTypes
            s = MODPF.pairwise_initiate(F, T)
            s_tmp = convert(T, get(pairwise_functions_initiate, f, 0))

            @test typeof(s) == T
            @test s == s_tmp

            f_tmp = get(pairwise_functions_aggregate, f, (s,x,y) -> convert(T, NaN))
            for x in (zero(T), one(T), convert(T,2)), y in (zero(T), one(T), convert(T,2))
                s = MODPF.pairwise_aggregate(F, one(T), x, y)
                s_tmp = f_tmp(one(T), x, y)

                @test typeof(s) == T
                @test s == s_tmp
            end

            f_tmp = get(pairwise_functions_return, f, s -> s)
            for x in (zero(T), one(T), convert(T,2))
                s = MODPF.pairwise_return(F, x)
                s_tmp = f_tmp(x)

                @test typeof(s) == T
                @test s == s_tmp
            end

            test_properties = get(pairwise_functions_properties, f, (false,false))
            @test MODPF.isstationary(F) == test_properties[1]
            @test MODPF.isisotropic(F)  == test_properties[2]
        end
    end
end
