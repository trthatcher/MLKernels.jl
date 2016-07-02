n = 30
m = 20
p = 5

test_pairwise_functions  = [(f_obj)() for f_obj in pairwise_functions]
test_composite_functions = [(f_obj)() for f_obj in composite_functions]
test_sample = [SquaredEuclidean(), ChiSquared(), ScalarProduct(), GaussianKernel()]

info("Testing ", MOD.unsafe_pairwise)
for T in FloatingPointTypes
    x = rand(T,p)
    y = rand(T,p)

    for f_test in test_pairwise_functions
        f = convert(RealFunction{T}, f_test)
        s = MOD.pairwise_initiate(f)
        for i in eachindex(y,x)
            s = MOD.pairwise_aggregate(f, s, x[i], y[i])
        end
        @test MOD.unsafe_pairwise(f, x, y) == MOD.pairwise_return(f, s)
    end

    for h_test in test_composite_functions
        h = convert(RealFunction{T}, h_test)
        s = MOD.composition(h.g, MOD.unsafe_pairwise(h.f, x, y))
        @test MOD.unsafe_pairwise(h, x, y) == s
    end

    for f_test in test_sample
        h = convert(RealFunction{T}, 2*f_test+1)
        s = h.a*MOD.unsafe_pairwise(h.f, x, y) + h.c
        @test MOD.unsafe_pairwise(h, x, y) == s
    end

    for scalar_op in (+, *), f_test1 in test_sample, f_test2 in test_sample
        h = convert(RealFunction{T}, (scalar_op)(f_test1, f_test2))
        s = (scalar_op)(MOD.unsafe_pairwise(h.f, x, y), MOD.unsafe_pairwise(h.g, x, y))
        @test MOD.unsafe_pairwise(h, x, y) == s

        h = convert(RealFunction{T}, (scalar_op)(2*f_test1+1, f_test2))
        s = (scalar_op)(MOD.unsafe_pairwise(h.f, x, y), MOD.unsafe_pairwise(h.g, x, y))
        @test MOD.unsafe_pairwise(h, x, y) == s
    end
end

info("Testing ", MOD.pairwise)
for T in FloatingPointTypes
    x = rand(T,p)
    y = rand(T,p)

    for f_test in test_pairwise_functions
        f = convert(RealFunction{T}, f_test)

        s1 = MOD.pairwise_return(f, MOD.pairwise_aggregate(f, MOD.pairwise_initiate(f), x[1], y[1]))
        @test MOD.pairwise(f, x[1], y[1]) == s1

        s2 = MOD.unsafe_pairwise(f, x, y)
        @test MOD.pairwise(f, x, y) == s2

        @test_throws DimensionMismatch MOD.pairwise(f, Array(T,p), Array(T,p+1))
        @test_throws DimensionMismatch MOD.pairwise(f, Array(T,p+1), Array(T,p))
    end

    for h_test in test_composite_functions
        h = convert(RealFunction{T}, h_test)
        
        @test MOD.pairwise(h, x[1], y[1]) == MOD.composition(h.g, MOD.pairwise(h.f, x[1], y[1]))
        @test MOD.pairwise(h, x, y) == MOD.composition(h.g, MOD.unsafe_pairwise(h.f, x, y))

        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p), Array(T,p+1))
        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p+1), Array(T,p))
    end

    for f_test in test_sample
        h = convert(RealFunction{T}, 2*f_test+1)

        s1 = h.a*MOD.pairwise(h.f, x[1], y[1]) + h.c
        @test MOD.pairwise(h, x[1], y[1]) == s1

        s2 = h.a*MOD.pairwise(h.f, x, y) + h.c
        @test MOD.pairwise(h, x, y) == s2

        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p), Array(T,p+1))
        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p+1), Array(T,p))
    end

    for scalar_op in (+, *), f_test1 in test_sample, f_test2 in test_sample
        h = convert(RealFunction{T}, (scalar_op)(f_test1, f_test2))

        s1 = (scalar_op)(MOD.pairwise(h.f, x[1], y[1]), MOD.pairwise(h.g, x[1], y[1]))
        @test MOD.pairwise(h, x[1], y[1]) == s1

        s2 = (scalar_op)(MOD.pairwise(h.f, x, y), MOD.pairwise(h.g, x, y))
        @test MOD.pairwise(h, x, y) == s2

        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p), Array(T,p+1))
        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p+1), Array(T,p))
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

info("Testing ", MOD.rectangular_compose!)
for f_obj in composition_classes
    for T in FloatingPointTypes
        X = rand(T, n, m)
        f = convert(MOD.CompositionClass{T}, (f_obj)())

        P = [MOD.composition(f,X[i,j]) for i in 1:size(X,1), j in 1:size(X,2)]
        @test_approx_eq MOD.rectangular_compose!(f, X)  P
    end
end

info("Testing ", MOD.symmetric_compose!)
for f_obj in composition_classes
    for T in FloatingPointTypes
        X = rand(T, n, n)
        f = convert(CompositionClass{T}, (f_obj)())

        P = LinAlg.copytri!([MOD.composition(f,X[i,j]) for i in 1:size(X,1), j in 1:size(X,2)], 'U')
        @test_approx_eq MOD.symmetric_compose!(f, X, true) P

        @test_throws DimensionMismatch MOD.symmetric_compose!(f, Array(T,n,n+1), true)
        @test_throws DimensionMismatch MOD.symmetric_compose!(f, Array(T,n+1,n), true)
    end
end

info("Testing ", MOD.pairwisematrix!)
test_set = (test_pairwise_functions..., test_composite_functions..., 
            [2*f+1 for f in test_sample]..., 
            [f1+f2 for f1 in test_sample, f2 in test_sample]...,
            [f1*f2 for f1 in test_sample, f2 in test_sample]...)
steps = length(test_set)
counter = 0
for f_test in test_set
    info("[", @sprintf("%3.0f", counter/steps*100), "%] Case ", @sprintf("%2.0f", counter+1), "/", steps)
    for T in FloatingPointTypes
        Set_X = [rand(T,p) for i = 1:n]
        Set_Y = [rand(T,p) for i = 1:m]

        X = transpose(hcat(Set_X...))
        Y = transpose(hcat(Set_Y...))
        f = convert(RealFunction{T}, f_test)
                   
        P = [MOD.pairwise(f,x,y) for x in Set_X, y in Set_X]
        @test_approx_eq MOD.pairwisematrix!(Val{:row}, Array(T,n,n), f, X,  true) P
        @test_approx_eq MOD.pairwisematrix!(Val{:col}, Array(T,n,n), f, X', true) P

        P = [MOD.pairwise(f,x,y) for x in Set_X, y in Set_Y]
        @test_approx_eq MOD.pairwisematrix!(Val{:row}, Array(T,n,m), f, X,  Y)  P
        @test_approx_eq MOD.pairwisematrix!(Val{:col}, Array(T,n,m), f, X', Y') P

        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n+1,n),   f, X, true)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n,n+1),   f, X, true)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n+1,n+1), f, X, true)

        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n+1,n),   f, X', true)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n,n+1),   f, X', true)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n+1,n+1), f, X', true)

        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n+1,m), f, X,  Y)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,n,m+1), f, X,  Y)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n+1,m), f, X', Y')
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,n,m+1), f, X', Y')

        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,m+1,n), f, Y,  X)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:row}, Array(T,m,n+1), f, Y,  X)
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,m+1,n), f, Y', X')
        @test_throws DimensionMismatch MOD.pairwisematrix!(Val{:col}, Array(T,m,n+1), f, Y', X')
    end
    counter += 1
end
info("[100%] Done")

info("Testing ", MOD.pairwisematrix)
test_set = test_sample
steps = length(test_set)
counter = 0
for f_test in test_set
    info("[", @sprintf("%3.0f", counter/steps*100), "%] Case ", @sprintf("%2.0f", counter+1), "/", steps)
    for T in FloatingPointTypes
        Set_X = [rand(T, p) for i = 1:n]
        Set_Y = [rand(T,p)  for i = 1:m]

        X = transpose(hcat(Set_X...))
        Y = transpose(hcat(Set_Y...))
        f = convert(RealFunction{T}, f_test)
                   
        P = [MOD.pairwise(f,x,y) for x in Set_X, y in Set_X]
        @test_approx_eq MOD.pairwisematrix(Val{:row}, f, X)  P
        @test_approx_eq MOD.pairwisematrix(Val{:col}, f, X') P
        @test_approx_eq MOD.pairwisematrix(f, X) P

        P = [MOD.pairwise(f,x,y) for x in Set_X, y in Set_Y]
        @test_approx_eq MOD.pairwisematrix(Val{:row}, f, X,  Y)  P
        @test_approx_eq MOD.pairwisematrix(Val{:col}, f, X', Y') P
        @test_approx_eq MOD.pairwisematrix(f, X, Y)  P

        @test_approx_eq f(Set_X[1][1], Set_Y[1][1]) MOD.pairwise(f, Set_X[1][1], Set_Y[1][1])
        @test_approx_eq f(Set_X[1],    Set_Y[1])    MOD.pairwise(f, Set_X[1],    Set_Y[1])
    end
    counter += 1
end
info("[100%] Done")
