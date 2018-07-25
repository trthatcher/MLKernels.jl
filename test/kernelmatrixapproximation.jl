n = 30
m = 20
p = 5

@info("Testing ", MOD.samplematrix)
X = Array{Int64}(n, m)
r = 0.15
@test length(MOD.samplematrix(RowMajor(), X, r)) == max(Int64(trunc(n*r)),1)
@test length(MOD.samplematrix(ColumnMajor(), X, r)) == max(Int64(trunc(m*r)),1)


@info("Testing ", MOD.nystrom_sample)
for T in FloatingPointTypes
    F = convert(Kernel{T}, GaussianKernel())
    X  = rand(T, n+5, p)
    Xs = X[1:n, :]

    for layout in (RowMajor(), ColumnMajor())
        X_tst, Xs_tst = layout == RowMajor() ? (X, Xs) : (permutedims(X), permutedims(Xs))

        K_tst = kernelmatrix(layout, F, Xs_tst, X_tst)
        Ks_tst = kernelmatrix(layout, F, Xs_tst, Xs_tst)

        K_tmp, Ks_tmp = MOD.nystrom_sample(layout, F, X_tst, [i for i = 1:n])

        @test isapprox(K_tmp,  K_tst)
        @test isapprox(Ks_tmp, Ks_tst)
    end
end

@info("Testing ", MOD.nystrom_pinv!)
for T in FloatingPointTypes
    X = rand(T, n, p)
    XtX = permutedims(X) * X

    @test isapprox(LinearAlgebra.pinv(XtX), MOD.nystrom_pinv!(copy(permutedims(X) * X)))
end

@info("Testing ", MOD.nystrom)
for T in FloatingPointTypes
    F = convert(Kernel{T}, GaussianKernel())
    X = rand(T, n+3, p)
    S = [i for i = 1:n]

    for layout in (RowMajor(), ColumnMajor())
        X_tst, Xs_tst = layout == RowMajor() ? (X, X[S,:]) : (permutedims(X), permutedims(X[S,:]))

        C_tst = permutedims(kernelmatrix(layout, F, X_tst, Xs_tst))
        W_tst = LinearAlgebra.pinv(kernelmatrix(layout, F, Xs_tst))

        KF = nystrom(layout, F, X_tst, S)

        @test isapprox(KF.C, C_tst)
        @test isapprox(KF.W, W_tst)
    end
end
