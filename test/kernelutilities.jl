n = 30
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
