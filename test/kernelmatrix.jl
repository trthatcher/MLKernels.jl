n = 30
m = 20
p = 5

@testset "Testing $(MLK.kernel)" begin
    for T in (Float32, Float64)
        x = rand(T,p)
        y = rand(T,p)

        x_alt = rand(T == Float32 ? Float64 : Float32, p)

        for f in kernel_functions
            P = (get(kernel_functions_base, f, SquaredEuclidean))()
            F = convert(f{T}, (f)())

            @test isapprox(MLK.kernel(F, x[1], y[1]), MLK.kappa(F, MLK.base_evaluate(P, x[1], y[1])))
            @test isapprox(MLK.kernel(F, x, y),       MLK.kappa(F, MLK.base_evaluate(P, x, y)))

            z = MLK.kernel(F, x_alt[1], y[1])
            @test typeof(z) == T

            z = MLK.kernel(F, x_alt, y)
            @test typeof(z) == T
        end
    end
end

@testset "Testing $(MLK.kappamatrix!)" begin
    for T in (Float32, Float64)
        X = rand(T,n,m)

        for f in kernel_functions
            F = convert(f{T}, (f)())

            K_tmp = [MLK.kappa(F,X[i]) for i in Base.Cartesian.CartesianIndices(size(X))]
            K_tst = MLK.kappamatrix!(F, copy(X))

            @test isapprox(K_tmp, K_tst)
        end
    end
end


@testset "Testing $(MLK.symmetric_kappamatrix!)" begin
    for T in (Float32, Float64)
        X = LinearAlgebra.copytri!(rand(T,n,n), 'U')

        for f in kernel_functions
            F = convert(f{T}, (f)())

            K_tmp = [MLK.kappa(F,X[i]) for i in Base.Cartesian.CartesianIndices(size(X))]
            K_tst = MLK.symmetric_kappamatrix!(F, copy(X), true)

            @test isapprox(K_tmp, K_tst)
            @test_throws DimensionMismatch MLK.symmetric_kappamatrix!(F, rand(T, p, p+1), true)
        end
    end
end


@testset "Testing $(MLK.kernelmatrix!)" begin
    for T in (Float32, Float64)
        X_set = [rand(T,p) for i = 1:n]
        Y_set = [rand(T,p) for i = 1:m]

        K_tst_nn = Array{T}(undef, n, n)
        K_tst_nm = Array{T}(undef, n, m)

        for layout in (Val(:row), Val(:col))
            X = layout == Val(:row) ? permutedims(hcat(X_set...)) : hcat(X_set...)
            Y = layout == Val(:row) ? permutedims(hcat(Y_set...)) : hcat(Y_set...)

            for f in kernel_functions
                F = convert(f{T}, (f)())

                K = [MLK.kernel(F,x,y) for x in X_set, y in X_set]
                @test isapprox(K, MLK.kernelmatrix!(layout, K_tst_nn, F, X, true))

                K = [MLK.kernel(F,x,y) for x in X_set, y in Y_set]
                @test isapprox(K, MLK.kernelmatrix!(layout, K_tst_nm, F, X, Y))
            end
        end
    end
end

@testset "Testing $(MLK.kernelmatrix)" begin
    for T in (Float32, Float64)
        X_set = [rand(Float32,p) for i = 1:n]
        Y_set = [rand(Float32,p) for i = 1:m]

        K_tst_nn = Array{T}(undef, n, n)
        K_tst_nm = Array{T}(undef, n, m)

        for layout in (Val(:row), Val(:col))
            isrowmajor = layout == Val(:row)
            X = convert(Array{T}, isrowmajor ? permutedims(hcat(X_set...)) : hcat(X_set...))
            Y = convert(Array{T}, isrowmajor ? permutedims(hcat(Y_set...)) : hcat(Y_set...))

            X_alt = convert(Array{T == Float32 ? Float64 : Float32}, X)
            Y_alt = convert(Array{T == Float32 ? Float64 : Float32}, Y)

            for f in kernel_functions
                F = convert(f{T}, (f)())

                K_tmp = MLK.kernelmatrix!(layout, K_tst_nn, F, X, true)

                K_tst = MLK.kernelmatrix(layout, F, X, true)
                @test isapprox(K_tmp, K_tst)
                @test eltype(K_tst) == T

                K_tst = MLK.kernelmatrix(layout, F, X_alt)
                @test isapprox(K_tmp, K_tst)
                @test eltype(K_tst) == T

                if isrowmajor
                    K_tst = MLK.kernelmatrix(F, X_alt)
                    @test isapprox(K_tmp, K_tst)
                    @test eltype(K_tst) == T
                end

                K_tmp = MLK.kernelmatrix!(layout, K_tst_nm, F, X, Y)

                K_tst = MLK.kernelmatrix(layout, F, X, Y)
                @test isapprox(K_tmp, K_tst)
                @test eltype(K_tst) == T

                K_tst = MLK.kernelmatrix(layout, F, X_alt, Y_alt)
                @test isapprox(K_tmp, K_tst)
                @test eltype(K_tst) == T

                if isrowmajor
                    K_tst = MLK.kernelmatrix(F, X_alt, Y_alt)
                    @test isapprox(K_tmp, K_tst)
                    @test eltype(K_tst) == T
                end
            end
        end
    end
end

@testset "Testing $(MLK.centerkernelmatrix!)" begin
    for T in FloatingPointTypes
        X = rand(T, n, p)
        Y = rand(T, m, p)

        K = X*permutedims(X)
        MLK.centerkernelmatrix!(K)

        Xc = X .- Statistics.mean(X, dims = 1)
        Kc = Xc*permutedims(Xc)

        @test isapprox(K, Kc)

        K = X*permutedims(Y)
        MLK.centerkernelmatrix!(K)

        Yc = Y .- Statistics.mean(Y, dims = 1)
        Kc = Xc*permutedims(Yc)

        @test isapprox(K, Kc)
    end
end
