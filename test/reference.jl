FloatingPointTypes = (Float32, Float64, BigFloat)

# Samples Input

x1 = [1; 2]
x2 = [2; 0]
x3 = [3; 2]
X = [x1';
     x2';
     x3']

y1 = [1; 1]
y2 = [1; 1]
Y = [y1'; 
     y2']

w = [2; 1]

Set_x = (x1,x2,x3)
Set_y = (y1,y2)

# Additive Kernel References

additive_kernels = (
    SquaredDistanceKernel,
    SineSquaredKernel,
    ChiSquaredKernel,
    ScalarProductKernel,
    MercerSigmoidKernel
)

additive_kernelfunctions = Dict(
    SquaredDistanceKernel => (t,x,y) -> ((x-y)^2)^t,
    SineSquaredKernel => (t,x,y) -> (sin(x-y)^2)^t,
    ChiSquaredKernel => (t,x,y) -> ((x-y)^2/(x+y))^t,
    ScalarProductKernel => (x,y) -> x*y,
    MercerSigmoidKernel => (d,b,x,y) -> tanh((x-d)/b) * tanh((y-d)/b)
)

additive_testinputs = Dict(
    SquaredDistanceKernel => ([1,1],[1,0],[0,1],[-1,-1],[-1,0],[0,-1]),
    SineSquaredKernel => ([1,1],[1,0],[0,1],[-1,-1],[-1,0],[0,-1]),
    ChiSquaredKernel => ([1,1],[1,0],[0,1]),
    ScalarProductKernel => ([1,1],[1,0],[0,1],[-1,-1],[-1,0],[0,-1]),
    MercerSigmoidKernel => ([1,1],[1,0],[0,1],[-1,-1],[-1,0],[0,-1])
)

additive_kernelargs = Dict(
    SquaredDistanceKernel => ([:t],[1],[0.5]),
    SineSquaredKernel => ([:t],[1],[0.5]),
    ChiSquaredKernel => ([:t],[1],[0.5]),
    ScalarProductKernel => (Symbol[], Int[], Int[]),
    MercerSigmoidKernel => ([:d,:b],[0,1],[0.5,2])
)

additive_ismercer = Dict(
    SquaredDistanceKernel => false,
    SineSquaredKernel => false,
    ChiSquaredKernel => false,
    ScalarProductKernel => true,
    MercerSigmoidKernel => true
)

additive_isnegdef = Dict(
    SquaredDistanceKernel => true,
    SineSquaredKernel => true,
    ChiSquaredKernel => true,
    ScalarProductKernel => false,
    MercerSigmoidKernel => false
)

additive_cases = Dict(
    SquaredDistanceKernel => ([0.25],[0.5],[1]),
    SineSquaredKernel => ([0.25],[0.5],[1]),
    ChiSquaredKernel => ([0.25],[0.5],[1]),
    ScalarProductKernel => (Int[],),
    MercerSigmoidKernel => ([0.5,2],[0,1])
)

additive_errorcases = Dict(
    SquaredDistanceKernel => ([1.1], [0]),
    SineSquaredKernel => ([1.1], [0]),
    ChiSquaredKernel => ([1.1], [0]),
    MercerSigmoidKernel => ([0,0],)
)



# Composite Kernel Cases

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
    ExponentialKernel => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    RationalQuadraticKernel => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    MaternKernel => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    PowerKernel => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    LogKernel => (SquaredDistanceKernel,SineSquaredKernel,ChiSquaredKernel),
    PolynomialKernel => (ScalarProductKernel,MercerSigmoidKernel),
    ExponentiatedKernel => (ScalarProductKernel,MercerSigmoidKernel),
    SigmoidKernel => (ScalarProductKernel,MercerSigmoidKernel)
)

composite_kernelfunctions = Dict(
    ExponentialKernel => (α,γ,z) -> exp(-α*z^γ),
    RationalQuadraticKernel => (α,β,γ,z) -> (1 + α*z^γ)^(-β),
    MaternKernel => (ν,θ,z) -> begin 
                                   v1 = sqrt(2*ν) * z / θ
                                   v1 = v1 < eps(typeof(z)) ? eps(typeof(z)) : v1
                                   2*(v1/2)^ν * besselk(ν,v1)/gamma(ν)
                               end,
    PowerKernel => (γ,z) -> z^γ,
    LogKernel => (α,γ,z) -> log(α*z^γ+1),
    PolynomialKernel => (α,c,d,z) -> (α*z+c)^d,
    ExponentiatedKernel => (α,z) -> exp(α*z),
    SigmoidKernel => (α,c,z) -> tanh(α*z+c)
)

composite_testinputs = Dict(
    ExponentialKernel => ([0],[1e-10],[0.5],[1]),
    RationalQuadraticKernel => ([0],[1e-10],[0.5],[1]),
    MaternKernel => ([0],[1e-10],[0.5],[1]),
    PowerKernel => ([0],[1e-10],[0.5],[1]),
    LogKernel => ([0],[1e-10],[0.5],[1]),
    PolynomialKernel => ([0],[1e-10],[0.5],[1],[-1e-10],[-0.5],[-1]),
    ExponentiatedKernel => ([0],[1e-10],[0.5],[1],[-1e-10],[-0.5],[-1]),
    SigmoidKernel => ([0],[1e-10],[0.5],[1],[-1e-10],[-0.5],[-1])
)

composite_kernelargs = Dict(
    ExponentialKernel => ([:alpha, :gamma], [1,1], [2,0.5]),
    RationalQuadraticKernel => ([:alpha, :beta, :gamma], [1,1,1], [2,2,0.5]),
    MaternKernel => ([:nu, :theta], [1,1], [2,2]),
    PowerKernel => ([:gamma], [1], [0.5]),
    LogKernel => ([:alpha, :gamma], [1,1], [2,0.5]),
    PolynomialKernel => ([:alpha, :c, :d], [1,1,2], [2,2,3]),
    ExponentiatedKernel => ([:alpha], [1], [2]),
    SigmoidKernel => ([:alpha, :c], [1, 1], [2, 2])
)

composite_ismercer = Dict(
    ExponentialKernel => true,
    RationalQuadraticKernel => true,
    MaternKernel => true,
    PowerKernel => false,
    LogKernel => false,
    PolynomialKernel => true,
    ExponentiatedKernel => true,
    SigmoidKernel => false
)

composite_isnegdef = Dict(
    ExponentialKernel => false,
    RationalQuadraticKernel => false,
    MaternKernel => false,
    PowerKernel => true,
    LogKernel => true,
    PolynomialKernel => false,
    ExponentiatedKernel => false,
    SigmoidKernel => false
)

composite_cases = Dict(
    ExponentialKernel => ([1,1], [1,0.5]),
    RationalQuadraticKernel => ([1,1,1], [1,1,0.5], [1,2,1], [2,2,0.5]),
    MaternKernel => ([1,1], [2,1]),
    PowerKernel => ([0.5], [1]),
    LogKernel => ([1,0.5], [1,1]),
    PolynomialKernel => ([1,1,2],[1,1,1]),
    ExponentiatedKernel => ([1],),
    SigmoidKernel => ([1,1],)
)

composite_errorcases = Dict(
    ExponentialKernel => ([0,1], [1,1.1], [1,0]),
    RationalQuadraticKernel => ([0,1,1], [1,0,1], [1,1,0], [1,1,1.1]),
    MaternKernel => ([1,0], [0,1]),
    PowerKernel => ([0], [1.1]),
    LogKernel => ([0,1], [1,0], [1,1.1]),
    PolynomialKernel => ([0,1,1],[1,-0.1,1],[1,1,1.5],[1,1,0]),
    ExponentiatedKernel => ([0],),
    SigmoidKernel => ([0,1],[1,-0.1])
)
