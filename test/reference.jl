# Additive Kernel References

composition_classes = (
    ExponentialClass,
    GammaExponentialClass,
    RationalClass,
    GammaRationalClass,
    MaternClass,
    ExponentiatedClass,
    PolynomialClass,
    PowerClass,
    LogClass,
    GammaLogClass,
    SigmoidClass
)

additive_kernels = (
    SquaredDistanceKernel,
    SineSquaredKernel,
    ChiSquaredKernel,
    ScalarProductKernel
)

composition_pairs = Dict(
    ExponentialClass      => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    GammaExponentialClass => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    RationalClass         => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    GammaRationalClass    => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    MaternClass           => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    ExponentiatedClass    => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    PolynomialClass       => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    PowerClass            => (ScalarProductKernel,),
    LogClass              => (ScalarProductKernel,),
    GammaLogClass         => (ScalarProductKernel,),
    SigmoidClass          => (ScalarProductKernel,)
)

all_default_args = Dict(
    SquaredDistanceKernel  => ([1], Int[]),
    SineSquaredKernel      => ([π,1], Int[]),
    ChiSquaredKernel       => ([1], Int[]),
    ScalarProductKernel    => (Int[], Int[]),
    GammaExponentialClass  => ([1,0.5],  Int[]),
    ExponentialClass       => ([1,], Int[]),
    GammaRationalClass     => ([1,1,0.5], Int[]),
    RationalClass          => ([1,1], Int[]),
    MaternClass            => ([1,1], Int[]),
    ExponentiatedClass     => ([1,0], Int[]),
    PolynomialClass        => ([1,1], Int[3]),
    PowerClass             => ([1,0,1], Int[]),
    GammaLogClass          => ([1,0.5], Int[]),
    LogClass               => ([1,], Int[]),
    SigmoidClass           => ([1,0], Int[])
)



#=
composition_kernels = (
    GaussianKernel,
    SquaredExponentialKernel,
    RadialBasisKernel,
    LaplacianKernel,
    PeriodicKernel,
    RationalKernel,
    MaternKernel,
    MatérnKernel,
    PolynomialKernel,
    LinearKernel,
    SigmoidKernel
)

all_test_args = Dict(
    SquaredDistanceKernel  => ([:t],[1],[0.5]),
    SineSquaredKernel      => ([:p, :t],[π,1],[2π,0.5]),
    ChiSquaredKernel       => ([:t],[1],[0.5]),
    ScalarProductKernel    => (Symbol[], Int[], Int[]),
    GammaExponentialClass  => ([:alpha, :gamma], [1,0.5], [2,1]),
    ExponentialClass       => ([:alpha], [1,], [2,]),
    GammaRationalClass     => ([:alpha, :beta, :gamma], [1,1,0.5], [2,2,1]),
    RationalClass          => ([:alpha, :beta], [1,1], [2,2]),
    MaternClass            => ([:nu, :theta], [1,1], [2,2]),
    ExponentiatedClass     => ([:a, :c], [1,0], [2,1]),
    PolynomialClass        => ([:a, :c, :d], [1,1,2], [2,2,3]),
    PowerClass             => ([:a, :c, :gamma], [1,0,1], [2,1,0.5]),
    GammaLogClass          => ([:alpha, :gamma], [1,0.5], [2,1]),
    LogClass               => ([:alpha], [1,], [2,]),
    SigmoidClass           => ([:a, :c], [1, 0], [2, 1])
)


all_kernelfunctions = Dict(
    SquaredDistanceKernel  => (t,x,y)   -> ((x-y)^2)^t,
    SineSquaredKernel      => (p,t,x,y) -> (sin(p*(x-y))^2)^t,
    ChiSquaredKernel       => (t,x,y)   -> (x == y == 0) ? zero(typeof(t)) : ((x-y)^2/(x+y))^t,
    ScalarProductKernel    => (x,y)     -> x*y,
    ExponentialClass       => (α,γ,z)   -> exp(-α*z^γ),
    RationalClass => (α,β,γ,z) -> (1 + α*z^γ)^(-β),
    MaternClass            => (ν,θ,z) -> begin 
                                              v1 = sqrt(2*ν) * z / θ
                                              v1 = v1 < eps(typeof(z)) ? eps(typeof(z)) : v1
                                              2*(v1/2)^ν * besselk(ν,v1)/gamma(ν)
                                          end,
    PowerClass             => (a,c,γ,z) -> (a*z+c)^γ,
    LogClass               => (α,γ,z)   -> log(α*z^γ+1),
    PolynomialClass        => (a,c,d,z) -> (a*z+c)^d,
    ExponentiatedClass     => (a,c,z)   -> exp(a*z+c),
    SigmoidClass           => (a,c,z)   -> tanh(a*z+c)

)

all_kernelproperties = Dict( #|atzero|atpos|atneg |mercer|negdef|
    SquaredDistanceKernel  => (true,  true, false, false, true), 
    SineSquaredKernel      => (true,  true, false, false, true),
    ChiSquaredKernel       => (true,  true, false, false, true),
    ScalarProductKernel    => (true,  true, true,  true,  false),
    ExponentialClass       => (false, true, false, true,  false),
    RationalClass => (false, true, false, true,  false),
    MaternClass            => (false, true, false, true,  false),
    PowerClass             => (true,  true, false, false, true),
    LogClass               => (true,  true, false, false, true),
    PolynomialClass        => (true,  true, true,  true,  false),
    ExponentiatedClass     => (false, true, false, true,  false),
    SigmoidClass           => (true,  true, true,  false, false)
)

all_testinputs = Dict(
    SquaredDistanceKernel  => ([1,1],[1,0],[0,1],[0,0],[-1,-1],[-1,0],[0,-1]),
    SineSquaredKernel      => ([1,1],[1,0],[0,1],[0,0],[-1,-1],[-1,0],[0,-1]),
    ChiSquaredKernel       => ([1,1],[1,0],[0,1],[0,0],[2,1],[1,2],[2,0],[0,2]),
    ScalarProductKernel    => ([1,1],[1,0],[0,1],[0,0],[-1,-1],[-1,0],[0,-1]),
    ExponentialClass       => ([0],[1e-10],[0.5],[1]),
    RationalClass => ([0],[1e-10],[0.5],[1]),
    MaternClass            => ([0],[1e-10],[0.5],[1]),
    PowerClass             => ([0],[1e-10],[0.5],[1]),
    LogClass               => ([0],[1e-10],[0.5],[1]),
    PolynomialClass        => ([0],[1e-10],[0.5],[1],[-1e-10],[-0.5],[-1]),
    ExponentiatedClass     => ([0],[1e-10],[0.5],[1],[-1e-10],[-0.5],[-1]),
    SigmoidClass           => ([0],[1e-10],[0.5],[1],[-1e-10],[-0.5],[-1])

)


all_phicases = Dict(
    SquaredDistanceKernel  => ([0.25],[0.5],[1]),
    SineSquaredKernel      => ([π,0.25],[π,0.5],[π,1]),
    ChiSquaredKernel       => ([0.25],[0.5],[1]),
    ScalarProductKernel    => (Int[],),
    ExponentialClass       => ([1,1], [1,0.5]),
    RationalClass => ([1,1,1], [1,1,0.5], [1,2,1], [2,2,0.5]),
    MaternClass            => ([1,1], [2,1]),
    PowerClass             => ([1,0,0.5], [1,0,1]),
    LogClass               => ([1,0.5], [1,1]),
    PolynomialClass        => ([1,1,2],[1,1,1]),
    ExponentiatedClass     => ([1,0],),
    SigmoidClass           => ([1,1],)
)

all_errorcases = Dict(
    SquaredDistanceKernel  => ([1.1], [0]),
    SineSquaredKernel      => ([π,1.1], [π,0], [0,1]),
    ChiSquaredKernel       => ([1.1], [0]),
    ExponentialClass       => ([0,1], [1,1.1], [1,0]),
    RationalClass => ([0,1,1], [1,0,1], [1,1,0], [1,1,1.1]),
    MaternClass            => ([1,0], [0,1]),
    PowerClass             => ([0,1,1], [0,-1,1], [1,0,0], [1,0,1.1]),
    LogClass               => ([0,1], [1,0], [1,1.1]),
    PolynomialClass        => ([0,1,1],[1,-0.1,1],[1,1,1.5],[1,1,0]),
    ExponentiatedClass     => ([0],),
    SigmoidClass           => ([0,1],[1,-0.1])
)

=#
