info("Testing ", MOD.PairwiseFunction)
for f_obj in pairwise_functions
    info("Testing ", f_obj)

    # Test constructors
    for T in FloatingPointTypes
        default_floats, default_others = pairwise_functions_defaults[f_obj]
        default_args = (T[default_floats...]..., default_others...)
        fields = fieldnames(f_obj)
        f = length(fields) == 0 ? (f_obj){T}() : (f_obj)(default_args...)

        @test eltype(f) == T

        if length(fields) == 0 && T == Float64
            @test eltype((f_obj)()) == T
        end

        for i in eachindex(fields)
            @test getfield(f, fields[i]).value == default_args[i]
        end

        s = MOD.pairwise_initiate(f)
        @test_approx_eq s pairwise_functions_initiate[f_obj]

        for (x,y) in ((zero(T),zero(T)), (zero(T),one(T)), (one(T),zero(T)), (one(T),one(T)))
            pw_agg = pairwise_functions_aggregate[f_obj]
            @test_approx_eq MOD.pairwise_aggregate(f, zero(T), x, y) pw_agg(zero(T), x, y, default_args...)
        end
    end

    # Test conversions
    for T in FloatingPointTypes
        f = (f_obj)()
        for U in FloatingPointTypes
            @test U == eltype(convert(RealFunction{U}, f))
        end
    end

    # Test Properties

    f = (f_obj)()

    properties = pairwise_functions_properties[f_obj]

    @test MOD.ismercer(f)        == properties[1]
    @test MOD.isnegdef(f)        == properties[2]
    @test MOD.ismetric(f)        == properties[3]
    @test MOD.isinnerprod(f)     == properties[4]
    @test MOD.attainsnegative(f) == properties[5]
    @test MOD.attainszero(f)     == properties[6]
    @test MOD.attainspositive(f) == properties[7]

    # Test that output does not create error
    show(DevNull, f)

end
