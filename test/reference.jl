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
    SquaredDistanceKernel => (t,x,y) -> (x-y)^(2t),
    SineSquaredKernel => (t,x,y) -> sin(x-y)^(2t),
    ChiSquaredKernel => (t,x,y) -> ((x-y)^2/(x+y))^t,
    ScalarProductKernel => (x,y) -> x*y,
    MercerSigmoidKernel => (d,b,x,y) -> tanh((x-d)/b) * tanh((y-d)/b)
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

composite_kernelfunctions = Dict(
    ExponentialKernel => phi(z,α,γ) = exp(-α*z^γ),
    RationalQuadraticKernel => phi(z,α,β,γ) = (1 + α*z^γ)^(-β),
    MaternKernel => phi(z,ν,θ) = 2*(sqrt(2*ν*z)/(2*θ))^ν * besselk(ν,z)/gamma(ν),
    PowerKernel => phi(z,γ) = z^γ,
    LogKernel => phi(z,α,γ) = log(α*z^γ+1),
    PolynomialKernel => phi(z,α,c,d) = (α*z+c)^d,
    ExponentiatedKernel => phi(z,α) = exp(α*z),
    SigmoidKernel => phi(z,α,c) = tanh(α*z+c)
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

composite_cases = Dict(
    ExponentialKernel => ([1,1], [1,0.5]),
    RationalQuadraticKernel => ([1,1,1], [1,1,0.5], [1,2,1]),
    MaternKernel => ([1,1,1], [1,2,1]),
    PowerKernel => ([0.5], [1]),
    LogKernel => ([1,0.5], [1,1]),
    PolynomialKernel => ([1,1,2],[1,1,2]),
    ExponentiatedKernel => ([1],),
    SigmoidKernel => ([1,1],)
)
