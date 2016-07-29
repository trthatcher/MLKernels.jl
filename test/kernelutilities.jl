n = 3
m = 20
p = 5

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

info("Testing ", MOD.kernelstatistics.env.name)
for T in FloatingPointTypes
    X = rand(T, n, n)
    x, s = MOD.kernelstatistics(X)

    @test_approx_eq x vec(mean(X,1))
    @test_approx_eq s mean(X)
end

info("Testing ", MOD.KernelCenterer.name.name)
for T in FloatingPointTypes
    K = rand(T, n, n)

    k, s = MOD.kernelstatistics(K)
    KC = MOD.KernelCenterer(K)

    @test_approx_eq k KC.mu_kappa
    @test_approx_eq s KC.mu_k
end

info("Testing ", MOD.center_symmetric!.env.name)
for T in FloatingPointTypes
    X = rand(T, n, p)
    K = X*transpose(X)
    KC = MOD.KernelCenterer(K)
    MOD.center_symmetric!(KC, K)

    Xc = X .- mean(X,1)
    Kc = Xc*transpose(Xc)

    @test_approx_eq K Kc
end
