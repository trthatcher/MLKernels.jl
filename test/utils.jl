n = 3
m = 2
p = 5

@testset "Testing $(MLK.promote_float)" begin
    @test MLK.promote_float() == Float64
    @test MLK.promote_float(Float16) == Float16
    @test MLK.promote_float(Float16, Float32) == Float32
    @test MLK.promote_float(Float16, Float32, Float64) == Float64
    @test MLK.promote_float(BigFloat, Float64, Float32, Float16) == BigFloat
    @test MLK.promote_float(Int64) == Float64
    @test MLK.promote_float(Int64, Float32) == Float32
end

@testset "Testing $(MLK.dotvectors!)" begin
    for T in FloatingPointTypes
        Set_X = [rand(T, p) for i = 1:n]
        Set_Y = [rand(T,p)  for i = 1:m]

        X = permutedims(hcat(Set_X...))
        Y = permutedims(hcat(Set_Y...))

        @test isapprox(MLK.dotvectors(Val(:row),    X), vec(sum((X.*X), dims = 2)))
        @test isapprox(MLK.dotvectors(Val(:col), X), vec(sum((X.*X), dims = 1)))

        @test isapprox(MLK.dotvectors!(Val(:row),    Vector{T}(undef, n), X), vec(sum((X.*X), dims = 2)))
        @test isapprox(MLK.dotvectors!(Val(:col), Vector{T}(undef, p), X), vec(sum((X.*X), dims = 1)))

        @test_throws DimensionMismatch MLK.dotvectors!(Val(:row),    Vector{T}(undef, 2), Array{T}(undef, 3,2))
        @test_throws DimensionMismatch MLK.dotvectors!(Val(:row),    Vector{T}(undef, 4), Array{T}(undef, 3,4))
        @test_throws DimensionMismatch MLK.dotvectors!(Val(:col), Vector{T}(undef, 2), Array{T}(undef, 2,3))
    end
end

@testset "Testing $(MLK.gramian!)" begin
    for T in FloatingPointTypes
        Set_X = [rand(T, p) for i = 1:n]
        Set_Y = [rand(T,p)  for i = 1:m]

        X = permutedims(hcat(Set_X...))
        Y = permutedims(hcat(Set_Y...))

        P = [LinearAlgebra.dot(x,y) for x in Set_X, y in Set_X]

        @test isapprox(MLK.gramian!(Val(:row),    Array{T}(undef, n,n), X,  true), P)
        @test isapprox(MLK.gramian!(Val(:col), Array{T}(undef, n,n), permutedims(X), true), P)

        P = [LinearAlgebra.dot(x,y) for x in Set_X, y in Set_Y]
        @test isapprox(MLK.gramian!(Val(:row),    Array{T}(undef, n,m), X,  Y),  P)
        @test isapprox(MLK.gramian!(Val(:col), Array{T}(undef, n,m), permutedims(X), permutedims(Y)), P)

        @test_throws DimensionMismatch MLK.gramian!(Val(:row), Array{T}(undef, n+1,n),   X, true)
        @test_throws DimensionMismatch MLK.gramian!(Val(:row), Array{T}(undef, n,n+1),   X, true)
        @test_throws DimensionMismatch MLK.gramian!(Val(:row), Array{T}(undef, n+1,n+1), X, true)

        @test_throws DimensionMismatch MLK.gramian!(Val(:col), Array{T}(undef, n+1,n),   permutedims(X), true)
        @test_throws DimensionMismatch MLK.gramian!(Val(:col), Array{T}(undef, n,n+1),   permutedims(X), true)
        @test_throws DimensionMismatch MLK.gramian!(Val(:col), Array{T}(undef, n+1,n+1), permutedims(X), true)

        @test_throws DimensionMismatch MLK.gramian!(Val(:row),    Array{T}(undef, n+1,m), X,  Y)
        @test_throws DimensionMismatch MLK.gramian!(Val(:row),    Array{T}(undef, n,m+1), X,  Y)
        @test_throws DimensionMismatch MLK.gramian!(Val(:col), Array{T}(undef, n+1,m), permutedims(X), permutedims(Y))
        @test_throws DimensionMismatch MLK.gramian!(Val(:col), Array{T}(undef, n,m+1), permutedims(X), permutedims(Y))

        @test_throws DimensionMismatch MLK.gramian!(Val(:row),    Array{T}(undef, m+1,n), Y,  X)
        @test_throws DimensionMismatch MLK.gramian!(Val(:row),    Array{T}(undef, m,n+1), Y,  X)
        @test_throws DimensionMismatch MLK.gramian!(Val(:col), Array{T}(undef, m+1,n), permutedims(Y), permutedims(X))
        @test_throws DimensionMismatch MLK.gramian!(Val(:col), Array{T}(undef, m,n+1), permutedims(Y), permutedims(X))
    end
end
