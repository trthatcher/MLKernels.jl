n = 200
m = 100
p = 25

info("Testing ", MOD.pairwise)
for kernelobj in additive_kernels
    for spX in (true,false), spY in (true,false)
        for T in FloatingPointTypes

            x = spX ? convert(SparseMatrixCSC{T},sprand(p,1,0.2)) : rand(T,p)
            y = spY ? convert(SparseMatrixCSC{T},sprand(p,1,0.2)) : rand(T,p)

            k = convert(Kernel{T}, (kernelobj)())
           
            @test MOD.pairwise(k, x[1], y[1]) == MOD.phi(k, x[1], y[1])
            @test MOD.pairwise(k, x, y)       == sum(map((x,y) -> MOD.phi(k,x,y), x, y))
        end
    end
end

info("Testing ", MOD.pairwisematrix)
for kernelobj in additive_kernels
    for spX in (true,false), spY in (true,false)
        for T in FloatingPointTypes

            Set_X = spX ? [convert(SparseMatrixCSC{T},sprand(p,1,0.2)) for i = 1:n] :
                          [rand(T, p) for i = 1:n]
            Set_Y = spY ? [convert(SparseMatrixCSC{T},sprand(p,1,0.2)) for i = 1:m] :
                          [rand(T,p) for i = 1:m]

            X = transpose(hcat(Set_X...))
            Y = transpose(hcat(Set_Y...))

            k = convert(Kernel{T}, (kernelobj)())
            #println("κ:", k, ", X:", typeof(X), ", Y:", typeof(Y))
           
            @test MOD.pairwise(k, Set_X[1][1], Set_Y[1][1]) == MOD.phi(k, Set_X[1][1], Set_Y[1][1])
            @test MOD.pairwise(k, Set_X[1], Set_Y[1]) == sum(map((x,y) -> MOD.phi(k,x,y), Set_X[1], Set_Y[1]))
            
            P = [MOD.pairwise(k,x,y) for x in Set_X, y in Set_X]
            @test_approx_eq MOD.pairwisematrix(Val{:row}, k, X)  P
            @test_approx_eq MOD.pairwisematrix(Val{:col}, k, X') P
            @test_approx_eq MOD.pairwisematrix!(Val{:row}, Array(T,n,n), k, X)  P
            @test_approx_eq MOD.pairwisematrix!(Val{:col}, Array(T,n,n), k, X') P

            P = [MOD.pairwise(k,x,y) for x in Set_X, y in Set_Y]
            @test_approx_eq MOD.pairwisematrix(Val{:row}, k, X,  Y)  P
            @test_approx_eq MOD.pairwisematrix(Val{:col}, k, X', Y') P
            @test_approx_eq MOD.pairwisematrix!(Val{:row}, Array(T,n,m), k, X,  Y)  P
            @test_approx_eq MOD.pairwisematrix!(Val{:col}, Array(T,n,m), k, X', Y') P

            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n+1,n), k, X)
            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n,n+1), k, X)
            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n+1,n), k, X')
            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n,n+1), k, X')

            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n+1,m), k, X,  Y)
            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n,m+1), k, X,  Y)
            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n+1,m), k, X', Y')
            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n,m+1), k, X', Y')

            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,m+1,n), k, Y,  X)
            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,m,n+1), k, Y,  X)
            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,m+1,n), k, Y', X')
            @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,m,n+1), k, Y', X')

            @test_approx_eq MOD.dotvectors(Val{:row}, X) sum((X.*X),2)
            @test_approx_eq MOD.dotvectors(Val{:col}, X) sum((X.*X),1)

            @test_approx_eq MOD.dotvectors!(Val{:row}, Array(T,n), X) sum((X.*X),2)
            @test_approx_eq MOD.dotvectors!(Val{:col}, Array(T,p), X) sum((X.*X),1)

            @test_throws DimensionMismatch MOD.dotvectors!(Val{:row}, Array(T,2), Array(T,3,2))
            @test_throws DimensionMismatch MOD.dotvectors!(Val{:row}, Array(T,4), Array(T,3,4))
            @test_throws DimensionMismatch MOD.dotvectors!(Val{:col}, Array(T,2), Array(T,2,3))
            @test_throws DimensionMismatch MOD.dotvectors!(Val{:col}, Array(T,4), Array(T,4,3))

            @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,3))
            @test_throws DimensionMismatch MOD.squared_distance!(Array(T,4,3), Array(T,3))

            @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,2), Array(T,4))
            @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,4), Array(T,4))
            @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,3), Array(T,3))
            @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,3), Array(T,5))
        end
    end
end
