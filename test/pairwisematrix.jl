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

info("Testing ", MOD.allocate_pairwisematrix)
for T in FloatingPointTypes
    X = rand(T,n,m)

    for layout in (RowMajor(), ColumnMajor())
        n_tmp = layout == RowMajor() ? n : m

        K = MOD.allocate_pairwisematrix(layout, X)
        @test size(K) == (n_tmp,n_tmp)
        @test eltype(K) == T
    end
end

info("Testing ", MOD.checkdimensions)
for layout in (RowMajor(), ColumnMajor())
    isrowmajor = layout == RowMajor()
    dim = isrowmajor ? 1 : 2

    X = isrowmajor ? Array(Float64,n,p) : Array(Float64,p,n)
    Y = isrowmajor ? Array(Float64,m,p) : Array(Float64,p,m)
    Y_bad = isrowmajor ? Array(Float64,m,p+1) : Array(Float64,p+1,m)

    @test MOD.checkdimensions(layout, Array(Float64,n,n), X) == n
    @test_throws DimensionMismatch MOD.checkdimensions(layout, Array(Float64,n,n+1), X)
    @test_throws DimensionMismatch MOD.checkdimensions(layout, Array(Float64,n+1,n+1), X)

    @test MOD.checkdimensions(layout, Array(Float64,n,m), X, Y) == (n,m)
    @test_throws DimensionMismatch MOD.checkdimensions(layout, Array(Float64,n,m+1), X, Y)
    @test_throws DimensionMismatch MOD.checkdimensions(layout, Array(Float64,n+1,m), X, Y)
    @test_throws DimensionMismatch MOD.checkdimensions(layout, Array(Float64,n,m), X, Y_bad)
end

info("Testing ", MOD.pairwisematrix!)
for T in (Float32, Float64)
    X_set = [rand(T,p) for i = 1:n]
    Y_set = [rand(T,p) for i = 1:m]

    P_tst_nn = Array(T, n, n)
    P_tst_nm = Array(T, n, m)

    for layout in (RowMajor(), ColumnMajor())
        X = layout == RowMajor() ? transpose(hcat(X_set...)) : hcat(X_set...)
        Y = layout == RowMajor() ? transpose(hcat(Y_set...)) : hcat(Y_set...)

        for f in pairwise_functions
            F = (f)()

            P = [MOD.pairwise(F,x,y) for x in X_set, y in X_set]
            @test_approx_eq P MOD.pairwisematrix!(layout, P_tst_nn, F, X, true)

            P = [MOD.pairwise(F,x,y) for x in X_set, y in Y_set]
            @test_approx_eq P MOD.pairwisematrix!(layout, P_tst_nm, F, X, Y)
        end
    end
end
