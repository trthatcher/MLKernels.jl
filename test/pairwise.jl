n = 200
m = 100
p = 25

info("Testing ", MOD.pairwise)
for kernelobj in additive_kernels
    for T in (Float64,)

        Set_X = [rand(T, p) for i = 1:n]
        Set_Y = [rand(T, p) for i = 1:m]

        X = transpose(hcat(Set_X...))
        Y = transpose(hcat(Set_Y...))

        k = (kernelobj)()
       
        @test MOD.pairwise(k, Set_X[1][1], Set_Y[1][1])     == MOD.phi(k, Set_X[1][1], Set_Y[1][1])
        @test MOD.pairwise(k, Set_X[1][1:2], Set_Y[1][1:2]) == MOD.phi(k, Set_X[1][1], Set_Y[1][1]) + MOD.phi(k, Set_X[1][2], Set_Y[1][2])
        
        P = [MOD.pairwise(k,x,y) for x in Set_X, y in Set_X]
        @test_approx_eq MOD.pairwise(Val{:row}, k, X)  P
        @test_approx_eq MOD.pairwise(Val{:col}, k, X') P
        @test_approx_eq MOD.pairwise!(Val{:row}, Array(T,n,n), k, X)  P
        @test_approx_eq MOD.pairwise!(Val{:col}, Array(T,n,n), k, X') P

        P = [MOD.pairwise(k,x,y) for x in Set_X, y in Set_Y]
        @test_approx_eq MOD.pairwise(Val{:row}, k, X,  Y)  P
        @test_approx_eq MOD.pairwise(Val{:col}, k, X', Y') P
        @test_approx_eq MOD.pairwise!(Val{:row}, Array(T,n,m), k, X,  Y)  P
        @test_approx_eq MOD.pairwise!(Val{:col}, Array(T,n,m), k, X', Y') P

        @test_throws DimensionMismatch MOD.pairwise!(Val{:row}, Array(T,n+1,n), k, X)
        @test_throws DimensionMismatch MOD.pairwise!(Val{:row}, Array(T,n,n+1), k, X)
        @test_throws DimensionMismatch MOD.pairwise!(Val{:col}, Array(T,n+1,n), k, X')
        @test_throws DimensionMismatch MOD.pairwise!(Val{:col}, Array(T,n,n+1), k, X')

        @test_throws DimensionMismatch MOD.pairwise!(Val{:row}, Array(T,n+1,m), k, X,  Y)
        @test_throws DimensionMismatch MOD.pairwise!(Val{:row}, Array(T,n,m+1), k, X,  Y)
        @test_throws DimensionMismatch MOD.pairwise!(Val{:col}, Array(T,n+1,m), k, X', Y')
        @test_throws DimensionMismatch MOD.pairwise!(Val{:col}, Array(T,n,m+1), k, X', Y')

        @test_throws DimensionMismatch MOD.pairwise!(Val{:row}, Array(T,m+1,n), k, Y,  X)
        @test_throws DimensionMismatch MOD.pairwise!(Val{:row}, Array(T,m,n+1), k, Y,  X)
        @test_throws DimensionMismatch MOD.pairwise!(Val{:col}, Array(T,m+1,n), k, Y', X')
        @test_throws DimensionMismatch MOD.pairwise!(Val{:col}, Array(T,m,n+1), k, Y', X')

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
