n = 30
m = 20
p = 5

@testset "Testing MODPF.unsafe_pairwise" begin
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
end

@testset "Testing MODPF.pairwise" begin
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
end

@testset "Testing MODPF.allocate_pairwisematrix" begin
    for T in FloatingPointTypes
        X = rand(T,n,m)

        for layout in (RowMajor(), ColumnMajor())
            n_tmp = layout == RowMajor() ? n : m

            K = MODPF.allocate_pairwisematrix(layout, X)
            @test size(K) == (n_tmp,n_tmp)
            @test eltype(K) == T
        end
    end
end

@testset "Testing MODPF.checkdimensions" begin
    for layout in (RowMajor(), ColumnMajor())
        isrowmajor = layout == RowMajor()
        dim = isrowmajor ? 1 : 2

        X =     isrowmajor ? Array{Float64}(undef, n,p)   : Array{Float64}(undef, p,n)
        Y =     isrowmajor ? Array{Float64}(undef, m,p)   : Array{Float64}(undef, p,m)
        Y_bad = isrowmajor ? Array{Float64}(undef, m,p+1) : Array{Float64}(undef, p+1,m)

        @test MODPF.checkdimensions(layout, Array{Float64}(undef, n,n), X) == n
        @test_throws DimensionMismatch MODPF.checkdimensions(layout, Array{Float64}(undef, n,n+1), X)
        @test_throws DimensionMismatch MODPF.checkdimensions(layout, Array{Float64}(undef, n+1,n+1), X)

        @test MODPF.checkdimensions(layout, Array{Float64}(undef, n,m), X, Y) == (n,m)
        @test_throws DimensionMismatch MODPF.checkdimensions(layout, Array{Float64}(undef, n,m+1), X, Y)
        @test_throws DimensionMismatch MODPF.checkdimensions(layout, Array{Float64}(undef, n+1,m), X, Y)
        @test_throws DimensionMismatch MODPF.checkdimensions(layout, Array{Float64}(undef, n,m), X, Y_bad)
    end
end

@testset "Testing MODPF.squared_distance!" begin
    for T in FloatingPointTypes
        Set_X = [rand(T,p) for i = 1:n]
        Set_Y = [rand(T,p) for i = 1:m]

        X = permutedims(hcat(Set_X...))
        Y = permutedims(hcat(Set_Y...))

        P = [LinearAlgebra.dot(x-y,x-y) for x in Set_X, y in Set_X]
        G = MODPF.gramian!(RowMajor(), Array{T}(undef, n,n), X, true)
        xtx = MODPF.dotvectors(RowMajor(), X)

        @test isapprox(MODPF.squared_distance!(G, xtx, true), P)

        P = [LinearAlgebra.dot(x-y,x-y) for x in Set_X, y in Set_Y]
        G = MODPF.gramian!(RowMajor(), Array{T}(undef, n,m), X, Y)
        xtx = MODPF.dotvectors(RowMajor(), X)
        yty = MODPF.dotvectors(RowMajor(), Y)
        MODPF.squared_distance!(G, xtx, yty)

        @test isapprox(G, P)
        @test all(G .>= 0)

        @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(undef, 3,4), Array{T}(undef, 3), true)
        @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(undef, 4,3), Array{T}(undef, 3), true)

        @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(undef, 3,4), Array{T}(undef, 2), Array{T}(undef, 4))
        @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(undef, 3,4), Array{T}(undef, 4), Array{T}(undef, 4))
        @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(undef, 3,4), Array{T}(undef, 3), Array{T}(undef, 3))
        @test_throws DimensionMismatch MODPF.squared_distance!(Array{T}(undef, 3,4), Array{T}(undef, 3), Array{T}(undef, 5))
    end
end

@testset "Testing MODPF.pairwisematrix!" begin
    for T in (Float32, Float64)
        X_set = [rand(T,p) for i = 1:n]
        Y_set = [rand(T,p) for i = 1:m]

        P_tst_nn = Array{T}(undef, n, n)
        P_tst_nm = Array{T}(undef, n, m)

        for layout in (RowMajor(), ColumnMajor())
            X = layout == RowMajor() ? permutedims(hcat(X_set...)) : hcat(X_set...)
            Y = layout == RowMajor() ? permutedims(hcat(Y_set...)) : hcat(Y_set...)

            for f in pairwise_functions
                F = (f)()

                P = [MODPF.pairwise(F,x,y) for x in X_set, y in X_set]
                @test isapprox(P, MODPF.pairwisematrix!(layout, P_tst_nn, F, X, true))

                P = [MODPF.pairwise(F,x,y) for x in X_set, y in Y_set]
                @test isapprox(P, MODPF.pairwisematrix!(layout, P_tst_nm, F, X, Y))
            end
        end
    end

    for layout in (RowMajor(), ColumnMajor())
        # Test case where squared_distance! may return a negative value
        v1 = [0.2585096115890490597877260370296426117420196533203125
              0.9692536801431554938091039730352349579334259033203125
              0.4741774214537994858176261914195492863655090332031250]

        v2 = [0.2585096115824996876320085448242025449872016906738281
              0.9692536801897344567180425656260922551155090332031250
              0.4741774214882835125628446348855504766106605529785156]

        v3 = rand(Float64, 3)

        F = SquaredEuclidean()
        X = layout == RowMajor() ? permutedims(hcat(v1, v2)) : hcat(v1, v2)
        Y = layout == RowMajor() ? permutedims(hcat(v1, v2, v3)) : hcat(v1, v2, v3)

        @test all(MODPF.pairwisematrix!(layout, Array{Float64}(undef, 2,2), F, X, true) .>= 0.0)
        @test all(MODPF.pairwisematrix!(layout, Array{Float64}(undef, 2,3), F, X, Y) .>= 0.0)
    end
end
