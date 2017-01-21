n = 30
m = 20
p = 5

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
