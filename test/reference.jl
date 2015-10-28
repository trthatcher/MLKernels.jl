FloatingPointTypes = (Float32, Float64, BigFloat)

# Additive Kernel References

additive_kernels = (
    SquaredDistanceKernel,
    SineSquaredKernel,
    ChiSquaredKernel,
    ScalarProductKernel,
    MercerSigmoidKernel
)

composite_kernels = (
    ExponentialKernel,
    RationalQuadraticKernel,
    MaternKernel,
    PowerKernel,
    LogKernel,
    PolynomialKernel,
    ExponentiatedKernel,
    SigmoidKernel
)

composite_pairs = Dict(
    ExponentialKernel       => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    RationalQuadraticKernel => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    MaternKernel            => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    PowerKernel             => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    LogKernel               => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    PolynomialKernel        => (ScalarProductKernel,MercerSigmoidKernel),
    ExponentiatedKernel     => (ScalarProductKernel,MercerSigmoidKernel),
    SigmoidKernel           => (ScalarProductKernel,MercerSigmoidKernel)
)

all_kernelargs = Dict(
    SquaredDistanceKernel   => ([:t],[1],[0.5]),
    SineSquaredKernel       => ([:t],[1],[0.5]),
    ChiSquaredKernel        => ([:t],[1],[0.5]),
    ScalarProductKernel     => (Symbol[], Int[], Int[]),
    MercerSigmoidKernel     => ([:d,:b],[0,1],[0.5,2]),
    ExponentialKernel       => ([:alpha, :gamma], [1,1], [2,0.5]),
    RationalQuadraticKernel => ([:alpha, :beta, :gamma], [1,1,1], [2,2,0.5]),
    MaternKernel            => ([:nu, :theta], [1,1], [2,2]),
    PowerKernel             => ([:gamma], [1], [0.5]),
    LogKernel               => ([:alpha, :gamma], [1,1], [2,0.5]),
    PolynomialKernel        => ([:alpha, :c, :d], [1,1,2], [2,2,3]),
    ExponentiatedKernel     => ([:alpha], [1], [2]),
    SigmoidKernel           => ([:alpha, :c], [1, 1], [2, 2])
)

all_kernelfunctions = Dict(
    SquaredDistanceKernel   => (t,x,y)   -> ((x-y)^2)^t,
    SineSquaredKernel       => (t,x,y)   -> (sin(x-y)^2)^t,
    ChiSquaredKernel        => (t,x,y)   -> (x == y == 0) ? zero(typeof(t)) : ((x-y)^2/(x+y))^t,
    ScalarProductKernel     => (x,y)     -> x*y,
    MercerSigmoidKernel     => (d,b,x,y) -> tanh((x-d)/b) * tanh((y-d)/b),
    ExponentialKernel       => (α,γ,z)   -> exp(-α*z^γ),
    RationalQuadraticKernel => (α,β,γ,z) -> (1 + α*z^γ)^(-β),
    MaternKernel            => (ν,θ,z) -> begin 
                                              v1 = sqrt(2*ν) * z / θ
                                              v1 = v1 < eps(typeof(z)) ? eps(typeof(z)) : v1
                                              2*(v1/2)^ν * besselk(ν,v1)/gamma(ν)
                                          end,
    PowerKernel             => (γ,z)     -> z^γ,
    LogKernel               => (α,γ,z)   -> log(α*z^γ+1),
    PolynomialKernel        => (α,c,d,z) -> (α*z+c)^d,
    ExponentiatedKernel     => (α,z)     -> exp(α*z),
    SigmoidKernel           => (α,c,z)   -> tanh(α*z+c)

)

all_kernelproperties = Dict( #|range|zero  |pos   |nonneg|mercer|negdef|
    SquaredDistanceKernel   => (:Rp, true,  false, true,  false, true), 
    SineSquaredKernel       => (:Rp, true,  false, true,  false, true),
    ChiSquaredKernel        => (:Rp, true,  false, true,  false, true),
    ScalarProductKernel     => (:R,  true,  false, false, true,  false),
    MercerSigmoidKernel     => (:R,  true,  false, false, true,  false),
    ExponentialKernel       => (:Rp, false, true,  true,  true,  false),
    RationalQuadraticKernel => (:Rp, false, true,  true,  true,  false),
    MaternKernel            => (:Rp, false, true,  true,  true,  false),
    PowerKernel             => (:Rp, true,  false, true,  false, true),
    LogKernel               => (:Rp, true,  false, true,  false, true),
    PolynomialKernel        => (:R,  true,  false, false, true,  false),
    ExponentiatedKernel     => (:Rp, false, true,  true,  true,  false),
    SigmoidKernel           => (:R,  true,  false, false, false, false)
)

all_testinputs = Dict(
    SquaredDistanceKernel   => ([1,1],[1,0],[0,1],[0,0],[-1,-1],[-1,0],[0,-1]),
    SineSquaredKernel       => ([1,1],[1,0],[0,1],[0,0],[-1,-1],[-1,0],[0,-1]),
    ChiSquaredKernel        => ([1,1],[1,0],[0,1],[0,0],[2,1],[1,2],[2,0],[0,2]),
    ScalarProductKernel     => ([1,1],[1,0],[0,1],[0,0],[-1,-1],[-1,0],[0,-1]),
    MercerSigmoidKernel     => ([1,1],[1,0],[0,1],[0,0],[-1,-1],[-1,0],[0,-1]),
    ExponentialKernel       => ([0],[1e-10],[0.5],[1]),
    RationalQuadraticKernel => ([0],[1e-10],[0.5],[1]),
    MaternKernel            => ([0],[1e-10],[0.5],[1]),
    PowerKernel             => ([0],[1e-10],[0.5],[1]),
    LogKernel               => ([0],[1e-10],[0.5],[1]),
    PolynomialKernel        => ([0],[1e-10],[0.5],[1],[-1e-10],[-0.5],[-1]),
    ExponentiatedKernel     => ([0],[1e-10],[0.5],[1],[-1e-10],[-0.5],[-1]),
    SigmoidKernel           => ([0],[1e-10],[0.5],[1],[-1e-10],[-0.5],[-1])

)


all_phicases = Dict(
    SquaredDistanceKernel   => ([0.25],[0.5],[1]),
    SineSquaredKernel       => ([0.25],[0.5],[1]),
    ChiSquaredKernel        => ([0.25],[0.5],[1]),
    ScalarProductKernel     => (Int[],),
    MercerSigmoidKernel     => ([0.5,2],[0,1]),
    ExponentialKernel       => ([1,1], [1,0.5]),
    RationalQuadraticKernel => ([1,1,1], [1,1,0.5], [1,2,1], [2,2,0.5]),
    MaternKernel            => ([1,1], [2,1]),
    PowerKernel             => ([0.5], [1]),
    LogKernel               => ([1,0.5], [1,1]),
    PolynomialKernel        => ([1,1,2],[1,1,1]),
    ExponentiatedKernel     => ([1],),
    SigmoidKernel           => ([1,1],)
)

all_errorcases = Dict(
    SquaredDistanceKernel   => ([1.1], [0]),
    SineSquaredKernel       => ([1.1], [0]),
    ChiSquaredKernel        => ([1.1], [0]),
    MercerSigmoidKernel     => ([0,0],),
    ExponentialKernel       => ([0,1], [1,1.1], [1,0]),
    RationalQuadraticKernel => ([0,1,1], [1,0,1], [1,1,0], [1,1,1.1]),
    MaternKernel            => ([1,0], [0,1]),
    PowerKernel             => ([0], [1.1]),
    LogKernel               => ([0,1], [1,0], [1,1.1]),
    PolynomialKernel        => ([0,1,1],[1,-0.1,1],[1,1,1.5],[1,1,0]),
    ExponentiatedKernel     => ([0],),
    SigmoidKernel           => ([0,1],[1,-0.1])
)
