n = 30
m = 20
p = 5

info("Testing ", MOD.samplematrix)
X = Array(Int64, n, m)
r = 0.15
@test length(MOD.samplematrix(RowMajor(), X, r)) == Int64(trunc(n*r))
@test length(MOD.samplematrix(ColumnMajor(), X, r)) == Int64(trunc(m*r))


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

#=
info("Testing ", MOD.nystrom)
for T in FloatingPointTypes
    X = rand(T, n, p)
    F = convert(Kernel{T}, GaussianKernel())

    K = kernelmatrix(RowMajor(), F, X)

    for layout in (RowMajor(), ColumnMajor())
        X_tst = layout == RowMajor ? X : X'

        W, C = MOD.nystrom(layout, F, X_tst, X_tst)
        N = (C'W)*C
        @test_approx_eq N K
    end

    W, C = MOD.nystrom(ColumnMajor(), F, X', X')
    N = (C'W)*C
    @test_approx_eq N K
end
=#
