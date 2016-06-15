info("Testing ", MOD.CompositionClass)
@test MOD.iscomposable(MLKTest.TestClass(), MLKTest.TestKernel()) == false
for class_obj in composition_classes
    info("Testing ", class_obj)

    # Test constructors
    for T in FloatingPointTypes
        default_floats, default_others = all_default_args[class_obj]
        default_args = (T[default_floats...]..., default_others...)
        fields = fieldnames(class_obj)
        k = (class_obj)(default_args...)
        @test eltype(k) == T

        for i in eachindex(fields)
            @test getfield(k, fields[i]).value == default_args[i]
        end

        for z in (zero(T),one(T))
            f = all_phifunctions[class_obj]
            @test_approx_eq MOD.phi(k, z) f(default_args..., z)
        end
    end

    # Test conversions
    for T in FloatingPointTypes
        phi = (class_obj)()
        for U in FloatingPointTypes
            @test U == eltype(convert(CompositionClass{U}, phi))
        end
    end

    # Test equality
    phi1 = (class_obj)()
    for class_obj2 in composition_classes
        phi2 = (class_obj2)()
        if class_obj == class_obj2
            @test phi1 == phi2
        else
            @test phi1 != phi2
        end
    end

    # Test properties
    properties = all_kernelproperties[class_obj]
    phi = (class_obj)()

    @test MOD.ismercer(phi)        == properties[1]
    @test MOD.isnegdef(phi)        == properties[2]
    @test MOD.attainsnegative(phi) == properties[3]
    @test MOD.attainszero(phi)     == properties[4]
    @test MOD.attainspositive(phi) == properties[5]

    # Test iscomposable() rules
    valid_kernel_objs = composition_pairs[class_obj]
    for kernel_obj in additive_kernels
        isvalid = get(valid_kernel_objs, kernel_obj, false)
        k = (kernel_obj)()
        @test MOD.iscomposable(phi, k) == isvalid
    end

    # Test that output does not create error
    show(DevNull, phi)
end


info("Testing ", MOD.KernelComposition)
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
