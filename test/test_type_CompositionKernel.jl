info("Testing ", MOD.KernelComposition)
for class_obj in composition_classes
    phi = (class_obj)()
    valid_kernel_objs = composition_pairs[class_obj]
    for kernel_obj in additive_kernels 
        kappa = (kernel_obj)()
        if get(valid_kernel_objs, kernel_obj, false)
            psi = KernelComposition(phi, kappa)
            @test MOD.ismercer(psi)        == MOD.ismercer(phi)
            @test MOD.isnegdef(psi)        == MOD.isnegdef(phi)
            @test MOD.attainsnegative(psi) == MOD.attainsnegative(phi)
            @test MOD.attainszero(psi)     == MOD.attainszero(phi)
            @test MOD.attainspositive(psi) == MOD.attainspositive(phi)
            show(DevNull, psi)
        else
            @test_throws ErrorException KernelComposition(phi,kappa)
        end
    end
end
