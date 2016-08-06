n = 30
m = 20
p = 5

test_pairwise_functions  = [(f_obj)() for f_obj in pairwise_functions]
test_composite_functions = [(f_obj)() for f_obj in composite_functions]
test_sample = [SquaredEuclidean(), ChiSquared(), ScalarProduct(), GaussianKernel()]

info("Testing ", MOD.unsafe_pairwise.env.name)
for T in FloatingPointTypes
    x = rand(T,p)
    y = rand(T,p)

    x_U = convert(Vector{Float64}, x)
    y_U = convert(Vector{Float32}, y)
    U = promote_type(T, Float64)

    for f_test in test_pairwise_functions
        f = convert(RealFunction{T}, f_test)
        s = MOD.pairwise_initiate(f)
        for i in eachindex(y,x)
            s = MOD.pairwise_aggregate(f, s, x[i], y[i])
        end
        @test MOD.unsafe_pairwise(f, x, y) == MOD.pairwise_return(f, s)

        @test typeof(MOD.unsafe_pairwise(f, x_U, y_U)) == U
    end

    for h_test in test_composite_functions
        h = convert(RealFunction{T}, h_test)
        s = MOD.composition(h.g, MOD.unsafe_pairwise(h.f, x, y))
        @test MOD.unsafe_pairwise(h, x, y) == s

        @test typeof(MOD.unsafe_pairwise(h, x_U, y_U)) == U
    end

    for f_test in test_sample
        h = convert(RealFunction{T}, 2*f_test+1)
        s = h.a*MOD.unsafe_pairwise(h.f, x, y) + h.c
        @test MOD.unsafe_pairwise(h, x, y) == s

        @test typeof(MOD.unsafe_pairwise(h, x_U, y_U)) == U
    end

    for scalar_op in (+, *), f_test1 in test_sample, f_test2 in test_sample
        h = convert(RealFunction{T}, (scalar_op)(f_test1, f_test2))
        s = (scalar_op)(MOD.unsafe_pairwise(h.f, x, y), MOD.unsafe_pairwise(h.g, x, y))
        @test MOD.unsafe_pairwise(h, x, y) == s

        @test typeof(MOD.unsafe_pairwise(h, x_U, y_U)) == U

        h = convert(RealFunction{T}, (scalar_op)(2*f_test1+1, f_test2))
        s = (scalar_op)(MOD.unsafe_pairwise(h.f, x, y), MOD.unsafe_pairwise(h.g, x, y))
        @test MOD.unsafe_pairwise(h, x, y) == s

        @test typeof(MOD.unsafe_pairwise(h, x_U, y_U)) == U
    end
end

info("Testing ", MOD.pairwise.env.name)
for T in FloatingPointTypes
    x = rand(T,p)
    y = rand(T,p)

    x_U = convert(Vector{Float64}, x)
    y_U = convert(Vector{Float32}, y)
    U = promote_type(T, Float64)

    for f_test in test_pairwise_functions
        f = convert(RealFunction{T}, f_test)

        s1 = MOD.pairwise_return(f, MOD.pairwise_aggregate(f, MOD.pairwise_initiate(f), x[1], y[1]))
        @test MOD.pairwise(f, x[1], y[1]) == s1
        @test f(x[1], y[1]) == s1
        @test typeof(MOD.pairwise(f, x_U[1], y_U[1])) == U

        s2 = MOD.unsafe_pairwise(f, x, y)
        @test MOD.pairwise(f, x, y) == s2
        @test f(x, y) == s2
        @test typeof(MOD.pairwise(f, x_U, y_U)) == U

        @test_throws DimensionMismatch MOD.pairwise(f, Array(T,p), Array(T,p+1))
        @test_throws DimensionMismatch MOD.pairwise(f, Array(T,p+1), Array(T,p))
    end

    for h_test in test_composite_functions
        h = convert(RealFunction{T}, h_test)
        
        @test MOD.pairwise(h, x[1], y[1]) == MOD.composition(h.g, MOD.pairwise(h.f, x[1], y[1]))
        @test h(x[1], y[1]) == MOD.pairwise(h, x[1], y[1])
        @test typeof(MOD.pairwise(h, x_U[1], y_U[1])) == U

        @test MOD.pairwise(h, x, y) == MOD.composition(h.g, MOD.unsafe_pairwise(h.f, x, y))
        @test h(x, y) == MOD.pairwise(h, x, y)
        @test typeof(MOD.pairwise(h, x_U, y_U)) == U

        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p), Array(T,p+1))
        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p+1), Array(T,p))
    end

    for f_test in test_sample
        h = convert(RealFunction{T}, 2*f_test+1)

        s1 = h.a*MOD.pairwise(h.f, x[1], y[1]) + h.c
        @test MOD.pairwise(h, x[1], y[1]) == s1
        @test h(x[1], y[1]) == s1
        @test typeof(MOD.pairwise(h, x_U[1], y_U[1])) == U

        s2 = h.a*MOD.pairwise(h.f, x, y) + h.c
        @test MOD.pairwise(h, x, y) == s2
        @test h(x, y) == s2
        @test typeof(MOD.pairwise(h, x_U, y_U)) == U

        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p), Array(T,p+1))
        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p+1), Array(T,p))
    end

    for scalar_op in (+, *), f_test1 in test_sample, f_test2 in test_sample
        h = convert(RealFunction{T}, (scalar_op)(f_test1, f_test2))

        s1 = (scalar_op)(MOD.pairwise(h.f, x[1], y[1]), MOD.pairwise(h.g, x[1], y[1]))
        @test MOD.pairwise(h, x[1], y[1]) == s1
        @test h(x[1], y[1]) == s1
        @test typeof(MOD.pairwise(h, x_U[1], y_U[1])) == U

        s2 = (scalar_op)(MOD.pairwise(h.f, x, y), MOD.pairwise(h.g, x, y))
        @test MOD.pairwise(h, x, y) == s2
        @test h(x, y) == s2
        @test typeof(MOD.pairwise(h, x_U, y_U)) == U

        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p), Array(T,p+1))
        @test_throws DimensionMismatch MOD.pairwise(h, Array(T,p+1), Array(T,p))
    end
end


info("Testing ", MOD.rectangular_compose!.env.name)
for f_obj in composition_classes
    for T in FloatingPointTypes
        X = rand(T, n, m)
        f = convert(MOD.CompositionClass{T}, (f_obj)())

        P = [MOD.composition(f,X[i,j]) for i in 1:size(X,1), j in 1:size(X,2)]
        @test_approx_eq MOD.rectangular_compose!(f, X)  P
    end
end

info("Testing ", MOD.symmetric_compose!.env.name)
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

info("Testing ", MOD.pairwisematrix!.env.name)
test_set = (test_pairwise_functions..., test_composite_functions..., 
            [2*f+1 for f in test_sample]..., 
            [f1+f2 for f1 in test_sample, f2 in test_sample]...,
            [f1*f2 for f1 in test_sample, f2 in test_sample]...)
steps = length(test_set)
counter = 0
for f_test in test_set
    info("[", @sprintf("%3.0f", counter/steps*100), "%] Case ", @sprintf("%2.0f", counter+1), "/",
         steps, " - ", test_print(f_test))
    for T in FloatingPointTypes
        Set_X = [rand(T,p) for i = 1:n]
        Set_Y = [rand(T,p) for i = 1:m]

        X = transpose(hcat(Set_X...))
        Y = transpose(hcat(Set_Y...))
        f = convert(RealFunction{T}, f_test)
                   
        P = T[MOD.pairwise(f,x,y) for x in Set_X, y in Set_X]

        P_test = MOD.pairwisematrix!(Val{:row}, Array(T,n,n), f, X,  true)
        @test_approx_eq P_test P
        @test eltype(P_test) == T

        P_test = MOD.pairwisematrix!(Val{:col}, Array(T,n,n), f, X', true)
        @test_approx_eq P_test P
        @test eltype(P_test) == T

        P = T[MOD.pairwise(f,x,y) for x in Set_X, y in Set_Y]

        P_test = MOD.pairwisematrix!(Val{:row}, Array(T,n,m), f, X, Y)
        @test_approx_eq P_test P
        @test eltype(P_test) == T

        P_test = MOD.pairwisematrix!(Val{:col}, Array(T,n,m), f, X', Y')
        @test_approx_eq P_test P
        @test eltype(P_test) == T

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

info("Testing ", MOD.pairwisematrix.env.name)
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

        X_U = convert(Matrix{T == Float64 ? Float32 : Float64}, X)
        Y_U = convert(Matrix{T == Float64 ? Float32 : Float64}, Y)
        U = promote_type(T == Float64 ? Float32 : Float64, T)
                   
        P = T[MOD.pairwise(f,x,y) for x in Set_X, y in Set_X]
        @test_approx_eq MOD.pairwisematrix(Val{:row}, f, X)  P
        @test_approx_eq MOD.pairwisematrix(Val{:col}, f, X') P
        @test_approx_eq MOD.pairwisematrix(f, X) P

        @test eltype(MOD.pairwisematrix(Val{:row}, f, X_U))  == U
        @test eltype(MOD.pairwisematrix(Val{:col}, f, X_U')) == U
        @test eltype(MOD.pairwisematrix(f, X_U)) == U

        P = T[MOD.pairwise(f,x,y) for x in Set_X, y in Set_Y]
        @test_approx_eq MOD.pairwisematrix(Val{:row}, f, X,  Y)  P
        @test_approx_eq MOD.pairwisematrix(Val{:col}, f, X', Y') P
        @test_approx_eq MOD.pairwisematrix(f, X, Y)  P

        @test eltype(MOD.pairwisematrix(Val{:row}, f, X_U, Y_U))   == U
        @test eltype(MOD.pairwisematrix(Val{:col}, f, X_U', Y_U')) == U
        @test eltype(MOD.pairwisematrix(f, X_U, Y_U)) == U
    end
    counter += 1
end
info("[100%] Done")
