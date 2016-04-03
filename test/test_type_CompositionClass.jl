info("Testing ", MOD.CompositionClass)
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
    @test MOD.iscomposable(phi, MLKTest.TestKernel()) == false
    for kernel_obj in additive_kernels
        isvalid = get(valid_kernel_objs, kernel_obj, false)
        k = (kernel_obj)()
        @test MOD.iscomposable(phi, k) == isvalid
    end

    # Test that output does not create error
    show(DevNull, phi)
end
