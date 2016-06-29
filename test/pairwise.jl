n = 200
m = 100
p = 25

info("Testing ", MOD.unsafe_pairwise)
for f_obj in pairwise_functions
    for T in FloatingPointTypes
        x = rand(T,p)
        y = rand(T,p)
        f = convert(RealFunction{T}, (f_obj)())
       
        s = MOD.pairwise_initiate(f)
        for i in eachindex(y,x)
            s = MOD.pairwise_aggregate(f, s, x[i], y[i])
        end
        @test MOD.unsafe_pairwise(f, x, y) == MOD.pairwise_return(f, s)
    end
end

info("Testing ", MOD.pairwise)
for f_obj in pairwise_functions
    for T in FloatingPointTypes
        f = convert(RealFunction{T}, (f_obj)())
        x = rand(T,p)
        y = rand(T,p)

        s1 = MOD.pairwise_return(f, MOD.pairwise_aggregate(f, MOD.pairwise_initiate(f), x[1], y[1]))
        @test MOD.pairwise(f, x[1], y[1]) == s1

        s2 = MOD.unsafe_pairwise(f, x, y)
        @test MOD.pairwise(f, x, y) == s2

        for g_obj in composition_classes
            g = convert(CompositionClass{T}, (g_obj)())
            if MOD.iscomposable(g,f)
                h = convert(RealFunction{T}, CompositeFunction(g,f))
                
                @test MOD.pairwise(h, x[1], y[1]) == MOD.composition(g, s1)
                @test MOD.pairwise(h, x, y) == MOD.composition(g, s2)
            end
        end
    end
end

info("Testing ", MOD.dotvectors!)
for T in FloatingPointTypes
    Set_X = [rand(T, p) for i = 1:n]
    Set_Y = [rand(T,p)  for i = 1:m]

    X = transpose(hcat(Set_X...))
    Y = transpose(hcat(Set_Y...))
    
    @test_approx_eq MOD.dotvectors(Val{:row}, X) sum((X.*X),2)
    @test_approx_eq MOD.dotvectors(Val{:col}, X) sum((X.*X),1)

    @test_approx_eq MOD.dotvectors!(Val{:row}, Array(T,n), X) sum((X.*X),2)
    @test_approx_eq MOD.dotvectors!(Val{:col}, Array(T,p), X) sum((X.*X),1)

    @test_throws DimensionMismatch MOD.dotvectors!(Val{:row}, Array(T,2), Array(T,3,2))
    @test_throws DimensionMismatch MOD.dotvectors!(Val{:row}, Array(T,4), Array(T,3,4))
    @test_throws DimensionMismatch MOD.dotvectors!(Val{:col}, Array(T,2), Array(T,2,3))

end

info("Testing ", MOD.gramian!)
for T in FloatingPointTypes
    Set_X = [rand(T, p) for i = 1:n]
    Set_Y = [rand(T,p)  for i = 1:m]

    X = transpose(hcat(Set_X...))
    Y = transpose(hcat(Set_Y...))

    P = [dot(x,y) for x in Set_X, y in Set_X]

    @test_approx_eq MOD.gramian!(Val{:row}, Array(T,n,n), X,  true) P
    @test_approx_eq MOD.gramian!(Val{:col}, Array(T,n,n), X', true) P

    P = [dot(x,y) for x in Set_X, y in Set_Y]
    @test_approx_eq MOD.gramian!(Val{:row}, Array(T,n,m), X,  Y)  P
    @test_approx_eq MOD.gramian!(Val{:col}, Array(T,n,m), X', Y') P

    @test_throws DimensionMismatch MOD.gramian!(Val{:row}, Array(T,n+1,n),   X, true)
    @test_throws DimensionMismatch MOD.gramian!(Val{:row}, Array(T,n,n+1),   X, true)
    @test_throws DimensionMismatch MOD.gramian!(Val{:row}, Array(T,n+1,n+1), X, true)

    @test_throws DimensionMismatch MOD.gramian!(Val{:col}, Array(T,n+1,n),   X', true)
    @test_throws DimensionMismatch MOD.gramian!(Val{:col}, Array(T,n,n+1),   X', true)
    @test_throws DimensionMismatch MOD.gramian!(Val{:col}, Array(T,n+1,n+1), X', true)

    @test_throws DimensionMismatch MOD.gramian!(Val{:row}, Array(T,n+1,m), X,  Y)
    @test_throws DimensionMismatch MOD.gramian!(Val{:row}, Array(T,n,m+1), X,  Y)
    @test_throws DimensionMismatch MOD.gramian!(Val{:col}, Array(T,n+1,m), X', Y')
    @test_throws DimensionMismatch MOD.gramian!(Val{:col}, Array(T,n,m+1), X', Y')

    @test_throws DimensionMismatch MOD.gramian!(Val{:row}, Array(T,m+1,n), Y,  X)
    @test_throws DimensionMismatch MOD.gramian!(Val{:row}, Array(T,m,n+1), Y,  X)
    @test_throws DimensionMismatch MOD.gramian!(Val{:col}, Array(T,m+1,n), Y', X')
    @test_throws DimensionMismatch MOD.gramian!(Val{:col}, Array(T,m,n+1), Y', X')

end


info("Testing ", MOD.squared_distance!)
for T in FloatingPointTypes
    Set_X = [rand(T,p) for i = 1:n]
    Set_Y = [rand(T,p) for i = 1:m]

    X = transpose(hcat(Set_X...))
    Y = transpose(hcat(Set_Y...))

    P = [dot(x-y,x-y) for x in Set_X, y in Set_X]
    G = MOD.gramian!(Val{:row}, Array(T,n,n), X, true)
    xtx = MOD.dotvectors(Val{:row}, X)

    @test_approx_eq MOD.squared_distance!(G, xtx, true) P

    P = [dot(x-y,x-y) for x in Set_X, y in Set_Y]
    G = MOD.gramian!(Val{:row}, Array(T,n,m), X, Y)
    xtx = MOD.dotvectors(Val{:row}, X)
    yty = MOD.dotvectors(Val{:row}, Y)

    @test_approx_eq MOD.squared_distance!(G, xtx, yty)  P
    
    @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,3), true)
    @test_throws DimensionMismatch MOD.squared_distance!(Array(T,4,3), Array(T,3), true)

    @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,2), Array(T,4))
    @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,4), Array(T,4))
    @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,3), Array(T,3))
    @test_throws DimensionMismatch MOD.squared_distance!(Array(T,3,4), Array(T,3), Array(T,5))
end


info("Testing ", MOD.pairwisematrix!)
steps = length(pairwise_functions) + length(composite_functions)
counter = 0
for f_obj in (pairwise_functions..., composite_functions...)
    counter += 1
    info("[", @sprintf("%3.0f", counter/steps*100), "%] ", f_obj)
end





#=

info("Testing ", MOD.pairwisematrix!)
steps = length(additive_kernels) + 1
counter = 0
info("    Progress:   0.0%")
for f_obj in (additive_kernels..., MLKTest.PairwiseTestKernel)
    for T in FloatingPointTypes
        Set_X = [rand(T, p) for i = 1:n]
        Set_Y = [rand(T,p)  for i = 1:m]

        X = transpose(hcat(Set_X...))
        Y = transpose(hcat(Set_Y...))
        k = convert(Kernel{T}, (f_obj)())
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
=#
