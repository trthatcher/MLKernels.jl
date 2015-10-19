additive_kernelfunctions = Dict(
    SquaredDistanceKernel => phi(x,y,t) = (x-y)^(2t),
    SineSquaredKernel => phi(x,y,t) = sin(x-y)^(2t),
    ChiSquaredKernel => phi(x,y,t) = ((x-y)^2/(x+y))^t,
    ScalarProductKernel => phi(x,y) = x*y,
    MercerSigmoidKernel => phi(x,y,d,b) = tanh((x-d)/b) * tanh((y-d)/b)
)

additive_kernelargs = Dict(
    SquaredDistanceKernel => [1],
    SineSquaredKernel => [1],
    ChiSquaredKernel => [1],
    ScalarProductKernel => Int[],
    MercerSigmoidKernel => [0,1]
)

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
