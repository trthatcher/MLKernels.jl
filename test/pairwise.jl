n = 200
m = 100
p = 25

info("Testing ", MOD.pairwise)
for kernelobj in additive_kernels
    for T in FloatingPointTypes
        x = rand(T,p)
        y = rand(T,p)
        k = convert(Kernel{T}, (kernelobj)())
       
        @test MOD.pairwise(k, x[1], y[1]) == MOD.phi(k, x[1], y[1])
        @test MOD.pairwise(k, x, y)       == sum(map((x,y) -> MOD.phi(k,x,y), x, y))
    end
end

info("Testing ", MOD.pairwisematrix!)
steps = length(additive_kernels) + 1
counter = 0
info("    Progress:   0.0%")
for kernelobj in (additive_kernels..., MLKTest.PairwiseTestKernel)
    for T in FloatingPointTypes
        Set_X = [rand(T, p) for i = 1:n]
        Set_Y = [rand(T,p)  for i = 1:m]

        X = transpose(hcat(Set_X...))
        Y = transpose(hcat(Set_Y...))
        k = convert(Kernel{T}, (kernelobj)())
        #println("κ:", k, ", X:", typeof(X), ", Y:", typeof(Y))
                   
        P = [MOD.pairwise(k,x,y) for x in Set_X, y in Set_X]
        #@test_approx_eq MOD.pairwisematrix(Val{:row}, k, X)  P
        #@test_approx_eq MOD.pairwisematrix(Val{:col}, k, X') P
        @test_approx_eq MOD.pairwisematrix!(Val{:row}, Array(T,n,n), k, X,  true)  P
        @test_approx_eq MOD.pairwisematrix!(Val{:col}, Array(T,n,n), k, X', true) P

        P = [MOD.pairwise(k,x,y) for x in Set_X, y in Set_Y]
        #@test_approx_eq MOD.pairwisematrix(Val{:row}, k, X,  Y)  P
        #@test_approx_eq MOD.pairwisematrix(Val{:col}, k, X', Y') P
        @test_approx_eq MOD.pairwisematrix!(Val{:row}, Array(T,n,m), k, X,  Y)  P
        @test_approx_eq MOD.pairwisematrix!(Val{:col}, Array(T,n,m), k, X', Y') P

        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n+1,n),   k, X, true)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n,n+1),   k, X, true)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n+1,n+1), k, X, true)

        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n+1,n),   k, X', true)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n,n+1),   k, X', true)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n+1,n+1), k, X', true)

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

        @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,3), true)
        @test_throws DimensionMismatch MOD.squared_distance!(Array(T,4,3), Array(T,3), true)

        @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,2), Array(T,4))
        @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,4), Array(T,4))
        @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,3), Array(T,3))
        @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,3), Array(T,5))
    end
    counter += 1
    info("    Progress: ", @sprintf("%5.1f", counter/steps*100), "%")
end
