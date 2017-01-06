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

composition_classes_defaults = Dict(
    ExponentialClass      => ([1,], Int[]),
    GammaExponentialClass => ([1,0.5],  Int[]),
    RationalClass         => ([1,1], Int[]),
    GammaRationalClass    => ([1,1,0.5], Int[]),
    MaternClass           => ([1,1], Int[]),
    ExponentiatedClass    => ([1,0], Int[]),
    PolynomialClass       => ([1,1], Int[3]),
    PowerClass            => ([1,0,1], Int[]),
    LogClass              => ([1,], Int[]),
    GammaLogClass         => ([1,0.5], Int[]),
    SigmoidClass          => ([1,0], Int[])
)

composition_classes_compose = Dict(
    ExponentialClass      => (α,z)     -> exp(-α*z^γ),
    GammaExponentialClass => (α,γ,z)   -> exp(-α*z^γ),
    RationalClass         => (α,β,z)   -> (1 + α*z)^(-β),
    GammaRationalClass    => (α,β,γ,z) -> (1 + α*z^γ)^(-β),
    MaternClass           => (ν,θ,z)   -> begin 
                                              v1 = sqrt(2*ν) * z / θ
                                              v1 = v1 < eps(typeof(z)) ? eps(typeof(z)) : v1
                                              2*(v1/2)^ν * besselk(ν,v1)/gamma(ν)
                                          end,
    ExponentiatedClass    => (a,c,z)   -> exp(a*z+c),
    PolynomialClass       => (a,c,d,z) -> (a*z+c)^d,
    PowerClass            => (a,c,γ,z) -> (a*z+c)^γ,
    LogClass              => (α,z)     -> log(α*z+1),
    GammaLogClass         => (α,γ,z)   -> log(α*z^γ+1),
    SigmoidClass          => (a,c,z)   -> tanh(a*z+c)
)

composition_rule = Dict(
    ExponentialClass      => f -> isnegdef(f) && isnonnegative(f),
    GammaExponentialClass => f -> isnegdef(f) && isnonnegative(f),
    RationalClass         => f -> isnegdef(f) && isnonnegative(f),
    GammaRationalClass    => f -> isnegdef(f) && isnonnegative(f),
    MaternClass           => f -> isnegdef(f) && isnonnegative(f),
    ExponentiatedClass    => f -> ismercer(f),
    PolynomialClass       => f -> ismercer(f),
    PowerClass            => f -> isnegdef(f) && isnonnegative(f),
    LogClass              => f -> isnegdef(f) && isnonnegative(f),
    GammaLogClass         => f -> isnegdef(f) && isnonnegative(f),
    SigmoidClass          => f -> ismercer(f)
)

composition_class_properties = Dict( 
                            #|mercer|negdef|metric|inprod|neg   |zero  |pos
    ExponentiatedClass =>    (true,  false, false, false, false, false, true),
    SigmoidClass =>          (false, false, false, false, true,  true,  true),
    ExponentialClass =>      (true,  false, false, false, false, false, true),
    GammaLogClass =>         (false, true,  false, false, false, true,  true),
    GammaRationalClass =>    (true,  false, false, false, false, false, true),
    LogClass =>              (false, true,  false, false, false, true,  true),
    MaternClass =>           (true,  false, false, false, false, false, true),
    PolynomialClass =>       (true,  false, false, false, true,  true,  true),
    PowerClass =>            (false, true,  false, false, false, true,  true),
    GammaExponentialClass => (true,  false, false, false, false, false, true),
    RationalClass =>         (true,  false, false, false, false, false, true)
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
