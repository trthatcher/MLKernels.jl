n = 3
m = 2
p = 5

@info("Testing ", MODPF.dotvectors!)
for T in FloatingPointTypes
    Set_X = [rand(T, p) for i = 1:n]
    Set_Y = [rand(T,p)  for i = 1:m]

    X = permutedims(hcat(Set_X...))
    Y = permutedims(hcat(Set_Y...))

    @test isapprox(MODPF.dotvectors(RowMajor(),    X), vec(sum((X.*X), dims = 2)))
    @test isapprox(MODPF.dotvectors(ColumnMajor(), X), vec(sum((X.*X), dims = 1)))

    @test isapprox(MODPF.dotvectors!(RowMajor(),    Vector{T}(undef, n), X), vec(sum((X.*X), dims = 2)))
    @test isapprox(MODPF.dotvectors!(ColumnMajor(), Vector{T}(undef, p), X), vec(sum((X.*X), dims = 1)))

    @test_throws DimensionMismatch MODPF.dotvectors!(RowMajor(),    Vector{T}(undef, 2), Array{T}(undef, 3,2))
    @test_throws DimensionMismatch MODPF.dotvectors!(RowMajor(),    Vector{T}(undef, 4), Array{T}(undef, 3,4))
    @test_throws DimensionMismatch MODPF.dotvectors!(ColumnMajor(), Vector{T}(undef, 2), Array{T}(undef, 2,3))
end

@info("Testing ", MODPF.gramian!)
for T in FloatingPointTypes
    Set_X = [rand(T, p) for i = 1:n]
    Set_Y = [rand(T,p)  for i = 1:m]

    X = permutedims(hcat(Set_X...))
    Y = permutedims(hcat(Set_Y...))

    P = [LinearAlgebra.dot(x,y) for x in Set_X, y in Set_X]

    @test isapprox(MODPF.gramian!(RowMajor(),    Array{T}(undef, n,n), X,  true), P)
    @test isapprox(MODPF.gramian!(ColumnMajor(), Array{T}(undef, n,n), permutedims(X), true), P)

    P = [LinearAlgebra.dot(x,y) for x in Set_X, y in Set_Y]
    @test isapprox(MODPF.gramian!(RowMajor(),    Array{T}(undef, n,m), X,  Y),  P)
    @test isapprox(MODPF.gramian!(ColumnMajor(), Array{T}(undef, n,m), permutedims(X), permutedims(Y)), P)

    @test_throws DimensionMismatch MODPF.gramian!(RowMajor(), Array{T}(undef, n+1,n),   X, true)
    @test_throws DimensionMismatch MODPF.gramian!(RowMajor(), Array{T}(undef, n,n+1),   X, true)
    @test_throws DimensionMismatch MODPF.gramian!(RowMajor(), Array{T}(undef, n+1,n+1), X, true)

    @test_throws DimensionMismatch MODPF.gramian!(ColumnMajor(), Array{T}(undef, n+1,n),   permutedims(X), true)
    @test_throws DimensionMismatch MODPF.gramian!(ColumnMajor(), Array{T}(undef, n,n+1),   permutedims(X), true)
    @test_throws DimensionMismatch MODPF.gramian!(ColumnMajor(), Array{T}(undef, n+1,n+1), permutedims(X), true)

    @test_throws DimensionMismatch MODPF.gramian!(RowMajor(),    Array{T}(undef, n+1,m), X,  Y)
    @test_throws DimensionMismatch MODPF.gramian!(RowMajor(),    Array{T}(undef, n,m+1), X,  Y)
    @test_throws DimensionMismatch MODPF.gramian!(ColumnMajor(), Array{T}(undef, n+1,m), permutedims(X), permutedims(Y))
    @test_throws DimensionMismatch MODPF.gramian!(ColumnMajor(), Array{T}(undef, n,m+1), permutedims(X), permutedims(Y))

    @test_throws DimensionMismatch MODPF.gramian!(RowMajor(),    Array{T}(undef, m+1,n), Y,  X)
    @test_throws DimensionMismatch MODPF.gramian!(RowMajor(),    Array{T}(undef, m,n+1), Y,  X)
    @test_throws DimensionMismatch MODPF.gramian!(ColumnMajor(), Array{T}(undef, m+1,n), permutedims(Y), permutedims(X))
    @test_throws DimensionMismatch MODPF.gramian!(ColumnMajor(), Array{T}(undef, m,n+1), permutedims(Y), permutedims(X))
end
