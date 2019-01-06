n = 30
m = 20
p = 5

@testset "Testing $(MLK.unsafe_base_evaluate)" begin
    for f in base_functions
        F = (f)()
        f_agg_tmp = get(base_functions_aggregate, f, (s,x,y) -> NaN)
        f_ret_tmp = get(base_functions_return, f, s -> s)

        for T in FloatingPointTypes
            x = rand(T,p)
            y = rand(T,p)

            s = zero(T)
            for i = 1:p
                s = f_agg_tmp(s, x[i], y[i])
            end
            s = f_ret_tmp(s)

            @test isapprox(s, MLK.unsafe_base_evaluate(F, x, y))
        end
    end
end

@testset "Testing $(MLK.base_evaluate)" begin
    for f in base_functions
        F = (f)()

        for T in FloatingPointTypes
            v = rand(T,p+1)
            x = rand(T,p)
            y = rand(T,p)

            @test isapprox(MLK.base_evaluate(F, x, y),       MLK.unsafe_base_evaluate(F, x, y))
            @test isapprox(MLK.base_evaluate(F, x[1], y[1]), MLK.unsafe_base_evaluate(F, x[1:1], y[1:1]))

            @test_throws DimensionMismatch MLK.base_evaluate(F, x, v)
            @test_throws DimensionMismatch MLK.base_evaluate(F, v, x)
            @test_throws DimensionMismatch MLK.base_evaluate(F, T[], T[])
        end
    end
end

@testset "Testing $(MLK.allocate_basematrix)" begin
    for T in FloatingPointTypes
        X = rand(T,n,m)

        for layout in (Val(:row), Val(:col))
            n_tmp = layout == Val(:row) ? n : m

            K = MLK.allocate_basematrix(layout, X)
            @test size(K) == (n_tmp,n_tmp)
            @test eltype(K) == T
        end
    end
end

@testset "Testing $(MLK.checkdimensions)" begin
    for layout in (Val(:row), Val(:col))
        isrowmajor = layout == Val(:row)
        dim = isrowmajor ? 1 : 2

        X =     isrowmajor ? Array{Float64}(undef, n,p)   : Array{Float64}(undef, p,n)
        Y =     isrowmajor ? Array{Float64}(undef, m,p)   : Array{Float64}(undef, p,m)
        Y_bad = isrowmajor ? Array{Float64}(undef, m,p+1) : Array{Float64}(undef, p+1,m)

        @test MLK.checkdimensions(layout, Array{Float64}(undef, n,n), X) == n
        @test_throws DimensionMismatch MLK.checkdimensions(layout, Array{Float64}(undef, n,n+1), X)
        @test_throws DimensionMismatch MLK.checkdimensions(layout, Array{Float64}(undef, n+1,n+1), X)

        @test MLK.checkdimensions(layout, Array{Float64}(undef, n,m), X, Y) == (n,m)
        @test_throws DimensionMismatch MLK.checkdimensions(layout, Array{Float64}(undef, n,m+1), X, Y)
        @test_throws DimensionMismatch MLK.checkdimensions(layout, Array{Float64}(undef, n+1,m), X, Y)
        @test_throws DimensionMismatch MLK.checkdimensions(layout, Array{Float64}(undef, n,m), X, Y_bad)
    end
end

@testset "Testing $(MLK.squared_distance!)" begin
    for T in FloatingPointTypes
        Set_X = [rand(T,p) for i = 1:n]
        Set_Y = [rand(T,p) for i = 1:m]

        X = permutedims(hcat(Set_X...))
        Y = permutedims(hcat(Set_Y...))

        P = [LinearAlgebra.dot(x-y,x-y) for x in Set_X, y in Set_X]
        G = MLK.gramian!(Val(:row), Array{T}(undef, n,n), X, true)
        xtx = MLK.dotvectors(Val(:row), X)

        @test isapprox(MLK.squared_distance!(G, xtx, true), P)

        P = [LinearAlgebra.dot(x-y,x-y) for x in Set_X, y in Set_Y]
        G = MLK.gramian!(Val(:row), Array{T}(undef, n,m), X, Y)
        xtx = MLK.dotvectors(Val(:row), X)
        yty = MLK.dotvectors(Val(:row), Y)
        MLK.squared_distance!(G, xtx, yty)

        @test isapprox(G, P)
        @test all(G .>= 0)

        @test_throws DimensionMismatch MLK.squared_distance!(Array{T}(undef, 3,4), Array{T}(undef, 3), true)
        @test_throws DimensionMismatch MLK.squared_distance!(Array{T}(undef, 4,3), Array{T}(undef, 3), true)

        @test_throws DimensionMismatch MLK.squared_distance!(Array{T}(undef, 3,4), Array{T}(undef, 2), Array{T}(undef, 4))
        @test_throws DimensionMismatch MLK.squared_distance!(Array{T}(undef, 3,4), Array{T}(undef, 4), Array{T}(undef, 4))
        @test_throws DimensionMismatch MLK.squared_distance!(Array{T}(undef, 3,4), Array{T}(undef, 3), Array{T}(undef, 3))
        @test_throws DimensionMismatch MLK.squared_distance!(Array{T}(undef, 3,4), Array{T}(undef, 3), Array{T}(undef, 5))
    end
end

@testset "Testing $(MLK.basematrix!)" begin
    for T in (Float32, Float64)
        X_set = [rand(T,p) for i = 1:n]
        Y_set = [rand(T,p) for i = 1:m]

        P_tst_nn = Array{T}(undef, n, n)
        P_tst_nm = Array{T}(undef, n, m)

        for f in base_functions
            F = (f)()

            #X = permutedims(hcat(X_set...))
            #Y = permutedims(hcat(Y_set...))
            #
            #P = [MLK.base_evaluate(F,x,y) for x in X_set, y in X_set]
            #@test isapprox(P, MLK.basematrix(F, X, true))
            #
            #P = [MLK.base_evaluate(F,x,y) for x in X_set, y in Y_set]
            #@test isapprox(P, MLK.basematrix(F, X, Y))

            for layout in (Val(:row), Val(:col))
                X = layout == Val(:row) ? permutedims(hcat(X_set...)) : hcat(X_set...)
                Y = layout == Val(:row) ? permutedims(hcat(Y_set...)) : hcat(Y_set...)

                P = [MLK.base_evaluate(F,x,y) for x in X_set, y in X_set]
                @test isapprox(P, MLK.basematrix!(layout, P_tst_nn, F, X, true))

                P = [MLK.base_evaluate(F,x,y) for x in X_set, y in Y_set]
                @test isapprox(P, MLK.basematrix!(layout, P_tst_nm, F, X, Y))
            end
        end
    end

    for layout in (Val(:row), Val(:col))
        # Test case where squared_distance! may return a negative value
        v1 = [0.2585096115890490597877260370296426117420196533203125
              0.9692536801431554938091039730352349579334259033203125
              0.4741774214537994858176261914195492863655090332031250]

        v2 = [0.2585096115824996876320085448242025449872016906738281
              0.9692536801897344567180425656260922551155090332031250
              0.4741774214882835125628446348855504766106605529785156]

        v3 = rand(Float64, 3)

        F = SquaredEuclidean()
        X = layout == Val(:row) ? permutedims(hcat(v1, v2)) : hcat(v1, v2)
        Y = layout == Val(:row) ? permutedims(hcat(v1, v2, v3)) : hcat(v1, v2, v3)

        @test all(MLK.basematrix!(layout, Array{Float64}(undef, 2,2), F, X, true) .>= 0.0)
        @test all(MLK.basematrix!(layout, Array{Float64}(undef, 2,3), F, X, Y) .>= 0.0)
    end
end
