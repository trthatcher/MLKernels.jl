info("Testing ", MOD.AdditiveKernel)
for kernel_obj in additive_kernels
    info("Testing ", kernel_obj)

    # Test constructors
    for T in FloatingPointTypes
        default_floats, default_others = all_default_args[kernel_obj]
        default_args = (T[default_floats...]..., default_others...)
        fields = fieldnames(kernel_obj)
        k = length(fields) == 0 ? (kernel_obj){T}() : (kernel_obj)(default_args...)

        @test eltype(k) == T

        if length(fields) == 0 && T == Float64
            @test eltype((kernel_obj)()) == T
        end

        for i in eachindex(fields)
            @test getfield(k, fields[i]).value == default_args[i]
        end

        for (x,y) in ((zero(T),zero(T)), (zero(T),one(T)), (one(T),zero(T)), (one(T),one(T)))
            f = all_phifunctions[kernel_obj]
            @test_approx_eq MOD.phi(k, x, y) f(default_args..., x, y)
        end
    end

    # Test conversions
    for T in FloatingPointTypes
        k = (kernel_obj)()
        for U in FloatingPointTypes
            @test U == eltype(convert(Kernel{U}, k))
        end
    end

    # Test Properties
    properties = all_kernelproperties[kernel_obj]
    k = (kernel_obj)()

    @test MOD.ismercer(k)        == properties[1]
    @test MOD.isnegdef(k)        == properties[2]
    @test MOD.attainsnegative(k) == properties[3]
    @test MOD.attainszero(k)     == properties[4]
    @test MOD.attainspositive(k) == properties[5]

    # Test that output does not create error
    show(DevNull, k)

end
