# Pairwise Function References

base_functions = (
    SquaredEuclidean,
    SineSquared,
    ChiSquared,
    ScalarProduct
)

base_functions_initiate = Dict(
    SquaredEuclidean => 0,
    SineSquared      => 0,
    ChiSquared       => 0,
    ScalarProduct    => 0
)

base_functions_aggregate = Dict(
    SquaredEuclidean => (s,x,y) -> s + (x-y)^2,
    SineSquared      => (s,x,y) -> s + sin((x-y))^2,
    ChiSquared       => (s,x,y) -> x == y == 0 ? s : s + ((x-y)^2/(x+y)),
    ScalarProduct    => (s,x,y) -> s + x*y
)

base_functions_return = Dict{DataType,Any}()


# Kernel Function References

kernel_functions = (
    ExponentialKernel,
    SquaredExponentialKernel,
    GammaExponentialKernel,
    RationalQuadraticKernel,
    GammaRationalKernel,
    MaternKernel,
    LinearKernel,
    PolynomialKernel,
    ExponentiatedKernel,
    PeriodicKernel,
    PowerKernel,
    LogKernel,
    SigmoidKernel
)

kernel_functions_arguments = Dict(
    ExponentialKernel        => ((1.0,),        (2.0,)),
    SquaredExponentialKernel => ((1.0,),        (2.0,)),
    GammaExponentialKernel   => ((1.0,1.0),     (2.0,0.5)),
    RationalQuadraticKernel  => ((1.0,1.0),     (2.0,2.0)),
    GammaRationalKernel      => ((1.0,1.0,1.0), (2.0,2.0,0.5)),
    MaternKernel             => ((1.0,1.0),     (2.0,2.0)),
    LinearKernel             => ((1.0,1.0),     (2.0,2.0)),
    PolynomialKernel         => ((1.0,1.0,3),   (2.0,2.0,2)),
    ExponentiatedKernel      => ((1.0,),        (2.0,)),
    PeriodicKernel           => ((1.0,),        (2.0,)),
    PowerKernel              => ((1.0,),        (0.5,)),
    LogKernel                => ((1.0,1.0),     (2.0,0.5)),
    SigmoidKernel            => ((1.0,1.0),     (2.0,2.0))
)

kernel_functions_kappa = Dict(
    ExponentialKernel        => (z,α)     -> exp(-α*sqrt(z)),
    SquaredExponentialKernel => (z,α)     -> exp(-α*z),
    GammaExponentialKernel   => (z,α,γ)   -> exp(-α*z^γ),
    RationalQuadraticKernel  => (z,α,β)   -> (1 + α*z)^(-β),
    GammaRationalKernel      => (z,α,β,γ) -> (1 + α*z^γ)^(-β),
    MaternKernel             => (z,ν,ρ)   -> begin
                                                d = √(z)
                                                T = typeof(z)
                                                d = d < eps(T) ? eps(T) : d
                                                tmp1 = √(2*ν)*d/ρ
                                                tmp2 = 2^(1 - ν)
                                                tmp2*(tmp1^ν)*besselk(ν, tmp1)/gamma(ν)
                                             end,
    LinearKernel             => (z,a,c)   -> (a*z+c),
    PolynomialKernel         => (z,a,c,d) -> (a*z+c)^d,
    ExponentiatedKernel      => (z,a)     -> exp(a*z),
    PeriodicKernel           => (z,α)     -> exp(-α*z),
    PowerKernel              => (z,γ)     -> z^γ,
    LogKernel                => (z,α,γ)   -> log(α*z^γ+1),
    SigmoidKernel            => (z,a,c)   -> tanh(a*z+c)
)


kernel_functions_base = Dict(
    LinearKernel             => ScalarProduct,
    PolynomialKernel         => ScalarProduct,
    ExponentiatedKernel      => ScalarProduct,
    PeriodicKernel           => SineSquared,
    SigmoidKernel            => ScalarProduct
)


base_functions_properties = Dict(
                       #|stnry |isotrop
    SineSquared      => (true,  false),
    ScalarProduct    => (false, false),
    SquaredEuclidean => (true,  true),
    ChiSquared       => (false, false)
 )
