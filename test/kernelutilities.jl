n = 30
m = 20
p = 5

info("Testing ", MOD.KernelCenterer.name.name)
for T in FloatingPointTypes
    K = rand(T, n, n)

    k = vec(mean(K,2))
    s = mean(k)
    KC = MOD.KernelCenterer(K)

    @test_approx_eq k KC.mux_kappa
    @test_approx_eq s KC.mu_kappa
end

info("Testing ", MOD.centerkernelmatrix!.env.name)
for T in FloatingPointTypes
    X = rand(T, n, p)
    Y = rand(T, m, p)

    K = X*transpose(X)
    MOD.centerkernelmatrix!(K, vec(mean(K,2)), vec(mean(K,1)), mean(K))

    Xc = X .- mean(X,1)
    Kc = Xc*transpose(Xc)

    @test_approx_eq K Kc

    K = X*transpose(Y)
    MOD.centerkernelmatrix!(K, vec(mean(K,2)), vec(mean(K,1)), mean(K))

    Yc = Y .- mean(Y,1)
    Kc = Xc*transpose(Yc)

    @test_approx_eq K Kc
end

info("Testing ", MOD.centerkernel!.env.name)
for T in FloatingPointTypes
    X = rand(T, n, p)

    K1 = X*transpose(X)
    K2 = copy(K1)

    Xc = X .- mean(X,1)
    Kc = Xc*transpose(Xc)

    KC = MOD.KernelCenterer(K1)

    MOD.centerkernel!(KC, K1)
    @test_approx_eq K1 Kc

    MOD.centerkernel!(K2)
    @test_approx_eq K2 Kc
end

info("Testing ", MOD.centerkernel.env.name)
for T in FloatingPointTypes
    X = rand(T, n, p)

    K1 = X*transpose(X)
    K2 = copy(K1)

    Xc = X .- mean(X,1)
    Kc = Xc*transpose(Xc)

    KC = MOD.KernelCenterer(K1)
    
    @test_approx_eq MOD.centerkernel(KC, K1) Kc
    @test_approx_eq K1 K2
    
    @test_approx_eq MOD.centerkernel(K1) Kc
    @test_approx_eq K1 K2
end

info("Testing ", MOD.KernelTransformer.name.name)
for T in FloatingPointTypes
    X = rand(T, n, p)
    Y = rand(T, m, p)
    k = convert(RealFunction{T}, ScalarProduct())

    KT = MOD.KernelTransformer(Val{:row}, k, X, true)
    @test KT.order == Val{:row}
    @test KT.kappa == k
    @test KT.X == X
    @test !(KT.X === X)
end

info("Testing ", MOD.nystrom.env.name)
for T in FloatingPointTypes
    X = rand(T, n, p)
    k = convert(RealFunction{T}, GaussianKernel(1/(2*p)))
    K = pairwisematrix(Val{:row}, k, X)

    W, C = MOD.nystrom(Val{:row}, k, X, Int64[i for i = 1:n])
    N = (C'W)*C
    @test_approx_eq N K

    W, C = MOD.nystrom(Val{:col}, k, X', Int64[i for i = 1:n])
    N = (C'W)*C
    @test_approx_eq N K
end
