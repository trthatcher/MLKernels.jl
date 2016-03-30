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

for kernel_obj in (GaussianKernel, LaplacianKernel, PeriodicKernel, RationalQuadraticKernel,
                   MaternKernel, PolynomialKernel, LinearKernel, SigmoidKernel)
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
