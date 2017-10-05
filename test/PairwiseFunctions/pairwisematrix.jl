n = 30
m = 20
p = 5

info("Testing ", MODPF.unsafe_pairwise)
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

        @test isapprox(s, MODPF.unsafe_pairwise(F, x, y))
    end
end

info("Testing ", MODPF.pairwise)
for f in pairwise_functions
    F = (f)()

    for T in FloatingPointTypes
        v = rand(T,p+1)
        x = rand(T,p)
        y = rand(T,p)

        @test isapprox(MODPF.pairwise(F, x, y),       MODPF.unsafe_pairwise(F, x, y))
        @test isapprox(MODPF.pairwise(F, x[1], y[1]), MODPF.unsafe_pairwise(F, x[1:1], y[1:1]))

        @test_throws DimensionMismatch MODPF.pairwise(F, x, v)
        @test_throws DimensionMismatch MODPF.pairwise(F, v, x)
        @test_throws DimensionMismatch MODPF.pairwise(F, T[], T[])
    end
end

info("Testing ", MODPF.allocate_pairwisematrix)
for T in FloatingPointTypes
    X = rand(T,n,m)

    for layout in (RowMajor(), ColumnMajor())
        n_tmp = layout == RowMajor() ? n : m

        K = MODPF.allocate_pairwisematrix(layout, X)
        @test size(K) == (n_tmp,n_tmp)
        @test eltype(K) == T
    end
end

info("Testing ", MODPF.checkdimensions)
for layout in (RowMajor(), ColumnMajor())
    isrowmajor = layout == RowMajor()
    dim = isrowmajor ? 1 : 2

    X =     isrowmajor ? Array{Float64}(n,p)   : Array{Float64}(p,n)
    Y =     isrowmajor ? Array{Float64}(m,p)   : Array{Float64}(p,m)
    Y_bad = isrowmajor ? Array{Float64}(m,p+1) : Array{Float64}(p+1,m)

    @test MODPF.checkdimensions(layout, Array{Float64}(n,n), X) == n
    @test_throws DimensionMismatch MODPF.checkdimensions(layout, Array{Float64}(n,n+1), X)
    @test_throws DimensionMismatch MODPF.checkdimensions(layout, Array{Float64}(n+1,n+1), X)

    @test MODPF.checkdimensions(layout, Array{Float64}(n,m), X, Y) == (n,m)
    @test_throws DimensionMismatch MODPF.checkdimensions(layout, Array{Float64}(n,m+1), X, Y)
    @test_throws DimensionMismatch MODPF.checkdimensions(layout, Array{Float64}(n+1,m), X, Y)
    @test_throws DimensionMismatch MODPF.checkdimensions(layout, Array{Float64}(n,m), X, Y_bad)
end

info("Testing ", MODPF.squared_distance!)
for T in FloatingPointTypes
    Set_X = [rand(T,p) for i = 1:n]
    Set_Y = [rand(T,p) for i = 1:m]

    X = transpose(hcat(Set_X...))
    Y = transpose(hcat(Set_Y...))

    P = [dot(x-y,x-y) for x in Set_X, y in Set_X]
    G = MODPF.gramian!(RowMajor(), Array{T}(n,n), X, true)
    xtx = MODPF.dotvectors(RowMajor(), X)

    @test isapprox(MODPF.squared_distance!(G, xtx, true), P)

    P = [dot(x-y,x-y) for x in Set_X, y in Set_Y]
    G = MODPF.gramian!(RowMajor(), Array{T}(n,m), X, Y)
    xtx = MODPF.dotvectors(RowMajor(), X)
    yty = MODPF.dotvectors(RowMajor(), Y)
    MODPF.squared_distance!(G, xtx, yty)

    @test isapprox(G, P)
    @test all(G .>= 0)
    
    @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(3,4), Array{T}(3), true)
    @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(4,3), Array{T}(3), true)

    @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(3,4), Array{T}(2), Array{T}(4))
    @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(3,4), Array{T}(4), Array{T}(4))
    @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(3,4), Array{T}(3), Array{T}(3))
    @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(3,4), Array{T}(3), Array{T}(5))
end

info("Testing ", MODPF.pairwisematrix!)
for T in (Float32, Float64)
    X_set = [rand(T,p) for i = 1:n]
    Y_set = [rand(T,p) for i = 1:m]

    P_tst_nn = Array{T}( n, n)
    P_tst_nm = Array{T}( n, m)

    for layout in (RowMajor(), ColumnMajor())
        X = layout == RowMajor() ? transpose(hcat(X_set...)) : hcat(X_set...)
        Y = layout == RowMajor() ? transpose(hcat(Y_set...)) : hcat(Y_set...)

        for f in pairwise_functions
            F = (f)()

            P = [MODPF.pairwise(F,x,y) for x in X_set, y in X_set]
            @test isapprox(P, MODPF.pairwisematrix!(layout, P_tst_nn, F, X, true))

            P = [MODPF.pairwise(F,x,y) for x in X_set, y in Y_set]
            @test isapprox(P, MODPF.pairwisematrix!(layout, P_tst_nm, F, X, Y))
        end
    end
end
