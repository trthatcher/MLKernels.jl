n = 30
m = 20
p = 5

info("Testing ", MOD.samplematrix)
X = Array(Int64, n, m)
r = 0.15
@test length(MOD.samplematrix(RowMajor(), X, r)) == max(Int64(trunc(n*r)),1)
@test length(MOD.samplematrix(ColumnMajor(), X, r)) == max(Int64(trunc(m*r)),1)


info("Testing ", MOD.nystrom_sample)
for T in FloatingPointTypes
    F = convert(Kernel{T}, GaussianKernel())
    X  = rand(T, n+5, p)
    Xs = X[1:n, :]

    for layout in (RowMajor(), ColumnMajor())
        X_tst, Xs_tst = layout == RowMajor() ? (X, Xs) : (X', Xs')
        
        K_tst = kernelmatrix(layout, F, Xs_tst, X_tst)
        Ks_tst = kernelmatrix(layout, F, Xs_tst, Xs_tst)
        
        K_tmp, Ks_tmp = MOD.nystrom_sample(layout, F, X_tst, [i for i = 1:n])

        @test_approx_eq K_tmp K_tst
        @test_approx_eq Ks_tmp Ks_tst
    end
end

info("Testing ", MOD.nystrom_pinv!)
for T in FloatingPointTypes
    X = rand(T, n, p)
    XtX = X'X

    @test_approx_eq pinv(XtX) MOD.nystrom_pinv!(copy(X'X))
end

info("Testing ", MOD.NystromFact)
for T in FloatingPointTypes
    F = convert(Kernel{T}, GaussianKernel())
    X = rand(T, n+3, p)
    S = [i for i = 1:n]

    for layout in (RowMajor(), ColumnMajor())
        X_tst, Xs_tst = layout == RowMajor() ? (X, X[S,:]) : (X', transpose(X[S,:]))

        C_tst = transpose(kernelmatrix(layout, F, X_tst, Xs_tst))
        W_tst = pinv(kernelmatrix(layout, F, Xs_tst))

        KF = NystromFact(layout, F, X_tst, S)

        @test_approx_eq KF.C C_tst
        @test_approx_eq KF.W W_tst
    end
end
