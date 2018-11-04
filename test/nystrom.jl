n = 30
m = 20
p = 5

@testset "Testing $(MLK.samplematrix)" begin
    X = Array{Int64}(undef, n, m)
    r = 0.15
    @test length(MLK.samplematrix(Val(:row), X, r)) == max(Int64(trunc(n*r)),1)
    @test length(MLK.samplematrix(Val(:col), X, r)) == max(Int64(trunc(m*r)),1)
end


@testset "Testing $(MLK.nystrom_sample)" begin
    for T in FloatingPointTypes
        F = convert(Kernel{T}, GaussianKernel())
        local X  = rand(T, n+5, p)
        Xs = X[1:n, :]

        for orientation in (Val(:row), Val(:col))
            X_tst, Xs_tst = orientation == Val(:row) ? (X, Xs) : (permutedims(X), permutedims(Xs))

            K_tst = kernelmatrix(orientation, F, Xs_tst, X_tst)
            Ks_tst = kernelmatrix(orientation, F, Xs_tst, Xs_tst)

            K_tmp, Ks_tmp = MLK.nystrom_sample(orientation, F, X_tst, [i for i = 1:n])

            @test isapprox(K_tmp,  K_tst)
            @test isapprox(Ks_tmp, Ks_tst)
        end
    end
end

@testset "Testing $(MLK.nystrom_pinv!)" begin
    for T in FloatingPointTypes
        local X = rand(T, n, p)
        XtX = permutedims(X) * X

        @test isapprox(LinearAlgebra.pinv(XtX), MLK.nystrom_pinv!(copy(permutedims(X) * X)))
    end
end

@testset "Testing $(MLK.nystrom)" begin
    for T in FloatingPointTypes
        F = convert(Kernel{T}, GaussianKernel())
        local X = rand(T, n+3, p)
        S = [i for i = 1:n]

        for orientation in (Val(:row), Val(:col))
            X_tst, Xs_tst = orientation == Val(:row) ? (X, X[S,:]) : (permutedims(X), permutedims(X[S,:]))

            C_tst = permutedims(kernelmatrix(orientation, F, X_tst, Xs_tst))
            W_tst = LinearAlgebra.pinv(kernelmatrix(orientation, F, Xs_tst))

            KF = nystrom(orientation, F, X_tst, S)

            @test isapprox(KF.C, C_tst)
            @test isapprox(KF.W, W_tst)
        end
    end
end
