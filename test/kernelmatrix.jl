n = 30
m = 20
p = 5

info("Testing ", MOD.kernel)
for T in (Float32, Float64)
    x = rand(T,p)
    y = rand(T,p)

    x_alt = rand(T == Float32 ? Float64 : Float32, p)

    for f in kernel_functions
        P = (get(kernel_functions_pairwise, f, SquaredEuclidean))()
        F = convert(f{T}, (f)())
        
        @test_approx_eq MOD.kernel(F, x[1], y[1]) MOD.kappa(F, MOD.pairwise(P, x[1], y[1]))
        @test_approx_eq MOD.kernel(F, x, y)       MOD.kappa(F, MOD.pairwise(P, x, y))

        z = MOD.kernel(F, x_alt[1], y[1])
        @test typeof(z) == T

        z = MOD.kernel(F, x_alt, y)
        @test typeof(z) == T
    end
end

info("Testing ", MOD.kappamatrix!)
for T in (Float32, Float64)
    X = rand(T,n,m)

    for f in kernel_functions
        F = convert(f{T}, (f)())

        K_tmp = [MOD.kappa(F,X[i]) for i in CartesianRange(size(X))]
        K_tst = MOD.kappamatrix!(F, copy(X))
        
        @test_approx_eq K_tmp K_tst
    end
end


info("Testing ", MOD.symmetric_kappamatrix!)
for T in (Float32, Float64)
    X = LinAlg.copytri!(rand(T,n,n), 'U')

    for f in kernel_functions
        F = convert(f{T}, (f)())

        K_tmp = [MOD.kappa(F,X[i]) for i in CartesianRange(size(X))]
        K_tst = MOD.symmetric_kappamatrix!(F, copy(X), true)
        
        @test_approx_eq K_tmp K_tst
        @test_throws DimensionMismatch MOD.symmetric_kappamatrix!(F, rand(T, p, p+1), true)
    end
end


info("Testing ", MOD.kernelmatrix!)
for T in (Float32, Float64)
    X_set = [rand(T,p) for i = 1:n]
    Y_set = [rand(T,p) for i = 1:m]

    K_tst_nn = Array(T, n, n)
    K_tst_nm = Array(T, n, m)

    for layout in (RowMajor(), ColumnMajor())
        X = layout == RowMajor() ? transpose(hcat(X_set...)) : hcat(X_set...)
        Y = layout == RowMajor() ? transpose(hcat(Y_set...)) : hcat(Y_set...)

        for f in kernel_functions
            F = convert(f{T}, (f)())

            K = [MOD.kernel(F,x,y) for x in X_set, y in X_set]
            @test_approx_eq K MOD.kernelmatrix!(layout, K_tst_nn, F, X, true)

            K = [MOD.kernel(F,x,y) for x in X_set, y in Y_set]
            @test_approx_eq K MOD.kernelmatrix!(layout, K_tst_nm, F, X, Y)
        end
    end
end

info("Testing ", MOD.kernelmatrix)
for T in (Float32, Float64)
    X_set = [rand(Float32,p) for i = 1:n]
    Y_set = [rand(Float32,p) for i = 1:m]

    K_tst_nn = Array(T, n, n)
    K_tst_nm = Array(T, n, m)

    for layout in (RowMajor(), ColumnMajor())
        isrowmajor = layout == RowMajor()
        X = convert(Array{T}, isrowmajor ? transpose(hcat(X_set...)) : hcat(X_set...))
        Y = convert(Array{T}, isrowmajor ? transpose(hcat(Y_set...)) : hcat(Y_set...))

        X_alt = convert(Array{T == Float32 ? Float64 : Float32}, X)
        Y_alt = convert(Array{T == Float32 ? Float64 : Float32}, Y)

        for f in kernel_functions
            F = convert(f{T}, (f)())

            K_tmp = MOD.kernelmatrix!(layout, K_tst_nn, F, X, true)

            K_tst = MOD.kernelmatrix(layout, F, X, true)
            @test_approx_eq K_tmp K_tst
            @test eltype(K_tst) == T

            K_tst = MOD.kernelmatrix(layout, F, X_alt)
            @test_approx_eq K_tmp K_tst
            @test eltype(K_tst) == T

            if isrowmajor
                K_tst = MOD.kernelmatrix(F, X_alt)
                @test_approx_eq K_tmp K_tst
                @test eltype(K_tst) == T
            end

            K_tmp = MOD.kernelmatrix!(layout, K_tst_nm, F, X, Y)

            K_tst = MOD.kernelmatrix(layout, F, X, Y)
            @test_approx_eq K_tmp K_tst
            @test eltype(K_tst) == T

            K_tst = MOD.kernelmatrix(layout, F, X_alt, Y_alt)
            @test_approx_eq K_tmp K_tst
            @test eltype(K_tst) == T

            if isrowmajor
                K_tst = MOD.kernelmatrix(F, X_alt, Y_alt)
                @test_approx_eq K_tmp K_tst
                @test eltype(K_tst) == T
            end
        end
    end
end

info("Testing ", MOD.centerkernelmatrix!)
for T in FloatingPointTypes
    X = rand(T, n, p)
    Y = rand(T, m, p)

    K = X*transpose(X)
    MOD.centerkernelmatrix!(K)

    Xc = X .- mean(X,1)
    Kc = Xc*transpose(Xc)

    @test_approx_eq K Kc

    K = X*transpose(Y)
    MOD.centerkernelmatrix!(K)

    Yc = Y .- mean(Y,1)
    Kc = Xc*transpose(Yc)

    @test_approx_eq K Kc
end
