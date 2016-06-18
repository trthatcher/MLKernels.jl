info("Testing ", MOD.CompositionClass)
#@test MOD.iscomposable(MLKTest.TestClass(), MLKTest.TestKernel()) == false
for class_obj in composition_classes
    info("Testing ", class_obj)

    # Test constructors
    for T in FloatingPointTypes
        default_floats, default_others = composition_classes_defaults[class_obj]
        default_args = (T[default_floats...]..., default_others...)
        fields = fieldnames(class_obj)
        g = (class_obj)(default_args...)
        @test eltype(g) == T

        for i in eachindex(fields)
            @test getfield(g, fields[i]).value == default_args[i]
        end

        for z in (zero(T),one(T))
            ft = composition_classes_compose[class_obj]
            @test_approx_eq MOD.composition(g, z) ft(default_args..., z)
        end
    end

    # Test conversions
    for T in FloatingPointTypes
        g = (class_obj)()
        for U in FloatingPointTypes
            @test U == eltype(convert(CompositionClass{U}, g))
        end
    end

    # Test equality
    g1 = (class_obj)()
    for class_obj2 in composition_classes
        g2 = (class_obj2)()
        if class_obj == class_obj2
            @test g1 == g2
        else
            @test g1 != g2
        end
    end

    # Test properties
    properties = composition_class_properties[class_obj]
    g = (class_obj)()

    @test MOD.ismercer(g)        == properties[1]
    @test MOD.isnegdef(g)        == properties[2]
    @test MOD.ismetric(g)        == properties[3]
    @test MOD.isinnerprod(g)     == properties[4]
    @test MOD.attainsnegative(g) == properties[5]
    @test MOD.attainszero(g)     == properties[6]
    @test MOD.attainspositive(g) == properties[7]

    # Test iscomposable() rules
    for f_obj in pairwise_functions
        f = (f_obj)()
        isvalid = get(composition_rule, class_obj, false)(f)
        @test MOD.iscomposable(g, f) == isvalid
    end

    # Test that output does not create error
    show(DevNull, g)
end

#=
info("Testing ", MOD.CompositeFunction)
for class_obj in composition_classes
    phi = (class_obj)()
    valid_kernel_objs = composition_pairs[class_obj]
    for kernel_obj in additive_kernels 
        kappa = (kernel_obj)()
        if get(valid_kernel_objs, kernel_obj, false)
            psi = phi ∘ kappa
            @test MOD.ismercer(psi)        == MOD.ismercer(phi)
            @test MOD.isnegdef(psi)        == MOD.isnegdef(phi)
            @test MOD.attainsnegative(psi) == MOD.attainsnegative(phi)
            @test MOD.attainszero(psi)     == MOD.attainszero(phi)
            @test MOD.attainspositive(psi) == MOD.attainspositive(phi)
            show(DevNull, psi)
        else
            @test_throws ErrorException phi ∘ kappa
        end
    end
end

for kernel_obj in composition_kernels
    @test typeof((kernel_obj)()) <: KernelComposition
end

for kernel_obj in additive_kernels
    kappa = (kernel_obj)()

    if MOD.iscomposable(PolynomialClass(), kappa)
        @test typeof((kappa^2).phi) <: PolynomialClass
    else
        @test_throws ErrorException kappa^2
    end

    if MOD.iscomposable(PowerClass(), kappa)
        @test typeof((kappa^0.5).phi) <: PowerClass
    else
        @test_throws ErrorException kappa^0.5
    end

    if MOD.iscomposable(ExponentiatedClass(), kappa)
        @test typeof((exp(kappa)).phi) <: ExponentiatedClass
    else
        @test_throws ErrorException exp(kappa)
    end

    if MOD.iscomposable(SigmoidClass(), kappa)
        @test typeof((tanh(kappa)).phi) <: SigmoidClass
    else
        @test_throws ErrorException tanh(kappa)
    end

end
=#
