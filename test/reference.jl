# Pairwise Function References

pairwise_functions = (
    Euclidean,
    SquaredEuclidean,
    SineSquared,
    ChiSquared,
    ScalarProduct
)

pairwise_functions_initiate = Dict(
    Euclidean        => 0,
    SquaredEuclidean => 0,
    SineSquared      => 0,
    ChiSquared       => 0,
    ScalarProduct    => 0
)

pairwise_functions_aggregate = Dict(
    Euclidean        => (s,x,y) -> s + (x-y)^2,
    SquaredEuclidean => (s,x,y) -> s + (x-y)^2,
    SineSquared      => (s,x,y) -> s + sin((x-y))^2,
    ChiSquared       => (s,x,y) -> x == y == 0 ? s : s + ((x-y)^2/(x+y)),
    ScalarProduct    => (s,x,y) -> s + x*y
)

pairwise_functions_return = Dict(
    Euclidean => s -> sqrt(s)
)



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
    ExponentialKernel        => (α,z)     -> exp(-α*sqrt(z)),
    SquaredExponentialKernel => (α,z)     -> exp(-α*z),
    GammaExponentialKernel   => (α,γ,z)   -> exp(-α*z^γ),
    RationalQuadraticKernel  => (α,β,z)   -> (1 + α*z)^(-β),
    GammaRationalKernel      => (α,β,γ,z) -> (1 + α*z^γ)^(-β),
    MaternKernel             => (ν,θ,z)   -> begin 
                                                v1 = sqrt(2*ν) * z / θ
                                                v1 = v1 < eps(typeof(z)) ? eps(typeof(z)) : v1
                                                2*(v1/2)^ν * besselk(ν,v1)/gamma(ν)
                                             end,
    LinearKernel             => (a,c,z)   -> (a*z+c),
    PolynomialKernel         => (a,c,d,z) -> (a*z+c)^d,
    ExponentiatedKernel      => (a,c,z)   -> exp(a*z+c),
    PeriodicKernel           => (α,z)     -> exp(-α*z),
    PowerKernel              => (a,c,γ,z) -> (a*z+c)^γ,
    LogKernel                => (α,γ,z)   -> log(α*z^γ+1),
    SigmoidKernel            => (a,c,z)   -> tanh(a*z+c)
)

#=
pairwise_functions_properties = Dict( 
                        #|mercer|negdef|metric|inprod|neg   |zero |pos
    SineSquaredKernel => (false, true,  false, false, false, true, true),
    ScalarProduct     => (true,  false, false, true,  true,  true, true),
    Euclidean         => (false, true,  true,  false, false, true, true),
    SquaredEuclidean  => (false, true,  true,  false, false, true, true),
    ChiSquared        => (false, true,  true,  false, false, true, true)
 )




composition_classes = (
    ExponentialKernel,
    GammaExponentialKernel,
    RationalKernel,
    GammaRationalKernel,
    MaternKernel,
    ExponentiatedKernel,
    PolynomialKernel,
    PowerKernel,
    LogKernel,
    GammaLogKernel,
    SigmoidKernel
)

composition_classes_defaults = Dict(
    ExponentialKernel      => ([1,], Int[]),
    GammaExponentialKernel => ([1,0.5],  Int[]),
    RationalKernel         => ([1,1], Int[]),
    GammaRationalKernel    => ([1,1,0.5], Int[]),
    MaternKernel           => ([1,1], Int[]),
    ExponentiatedKernel    => ([1,0], Int[]),
    PolynomialKernel       => ([1,1], Int[3]),
    PowerKernel            => ([1,0,1], Int[]),
    LogKernel              => ([1,], Int[]),
    GammaLogKernel         => ([1,0.5], Int[]),
    SigmoidKernel          => ([1,0], Int[])
)

composition_classes_compose = Dict(
    ExponentialKernel      => (α,z)     -> exp(-α*z^γ),
    GammaExponentialKernel => (α,γ,z)   -> exp(-α*z^γ),
    RationalKernel         => (α,β,z)   -> (1 + α*z)^(-β),
    GammaRationalKernel    => (α,β,γ,z) -> (1 + α*z^γ)^(-β),
    MaternKernel           => (ν,θ,z)   -> begin 
                                              v1 = sqrt(2*ν) * z / θ
                                              v1 = v1 < eps(typeof(z)) ? eps(typeof(z)) : v1
                                              2*(v1/2)^ν * besselk(ν,v1)/gamma(ν)
                                          end,
    ExponentiatedKernel    => (a,c,z)   -> exp(a*z+c),
    PolynomialKernel       => (a,c,d,z) -> (a*z+c)^d,
    PowerKernel            => (a,c,γ,z) -> (a*z+c)^γ,
    LogKernel              => (α,z)     -> log(α*z+1),
    GammaLogKernel         => (α,γ,z)   -> log(α*z^γ+1),
    SigmoidKernel          => (a,c,z)   -> tanh(a*z+c)
)

composition_rule = Dict(
    ExponentialKernel      => f -> isnegdef(f) && isnonnegative(f),
    GammaExponentialKernel => f -> isnegdef(f) && isnonnegative(f),
    RationalKernel         => f -> isnegdef(f) && isnonnegative(f),
    GammaRationalKernel    => f -> isnegdef(f) && isnonnegative(f),
    MaternKernel           => f -> isnegdef(f) && isnonnegative(f),
    ExponentiatedKernel    => f -> ismercer(f),
    PolynomialKernel       => f -> ismercer(f),
    PowerKernel            => f -> isnegdef(f) && isnonnegative(f),
    LogKernel              => f -> isnegdef(f) && isnonnegative(f),
    GammaLogKernel         => f -> isnegdef(f) && isnonnegative(f),
    SigmoidKernel          => f -> ismercer(f)
)

composition_class_properties = Dict( 
                            #|mercer|negdef|metric|inprod|neg   |zero  |pos
    ExponentiatedKernel =>    (true,  false, false, false, false, false, true),
    SigmoidKernel =>          (false, false, false, false, true,  true,  true),
    ExponentialKernel =>      (true,  false, false, false, false, false, true),
    GammaLogKernel =>         (false, true,  false, false, false, true,  true),
    GammaRationalKernel =>    (true,  false, false, false, false, false, true),
    LogKernel =>              (false, true,  false, false, false, true,  true),
    MaternKernel =>           (true,  false, false, false, false, false, true),
    PolynomialKernel =>       (true,  false, false, false, true,  true,  true),
    PowerKernel =>            (false, true,  false, false, false, true,  true),
    GammaExponentialKernel => (true,  false, false, false, false, false, true),
    RationalKernel =>         (true,  false, false, false, false, false, true)
 )

composite_functions = (
    GaussianKernel,
    SquaredExponentialKernel,
    RadialBasisKernel,
    LaplacianKernel,
    PeriodicKernel,
    RationalQuadraticKernel,
    MaternKernel,
    MatérnKernel,
    PolynomialKernel,
    LinearKernel,
    SigmoidKernel
)
=#
