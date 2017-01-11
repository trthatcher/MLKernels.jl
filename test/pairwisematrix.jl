n = 30
m = 20
p = 5

info("Testing ", MOD.unsafe_pairwise)
for f in pairwise_functions
    F = (f)()
    f_agg_tmp = get(pairwise_functions_aggregate, f, (s,x,y) -> NaN)
    f_ret_tmp = get(pairwise_functions_return, f, s -> s)

    for T in FloatingPointTypes
        x = rand(T,p)
        y = rand(T,p)

        s = zero(T)
        for i = 1:p
            s = f_agg_tmp(s, x[i], y[i])
        end
        s = f_ret_tmp(s)

        @test_approx_eq s MOD.unsafe_pairwise(F, x, y)
    end
end

info("Testing ", MOD.pairwise)
for f in pairwise_functions
    F = (f)()

    for T in FloatingPointTypes
        v = rand(T,p+1)
        x = rand(T,p)
        y = rand(T,p)

        @test_approx_eq MOD.pairwise(F, x, y) MOD.unsafe_pairwise(F, x, y)
        @test_approx_eq MOD.pairwise(F, x[1], y[1]) MOD.unsafe_pairwise(F, x[1:1], y[1:1])

        @test_throws DimensionMismatch MOD.pairwise(F, x, v)
        @test_throws DimensionMismatch MOD.pairwise(F, v, x)
        @test_throws DimensionMismatch MOD.pairwise(F, T[], T[])
    end
end
