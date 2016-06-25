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

info("Testing ", MOD.CompositeFunction)
for class_obj in composition_classes
    g = (class_obj)()
    for f_obj in pairwise_functions
        f = (f_obj)()
        isvalid = get(composition_rule, class_obj, false)(f)
        if isvalid
            h = g ∘ f
            @test MOD.ismercer(h)        == MOD.ismercer(g)
            @test MOD.isnegdef(h)        == MOD.isnegdef(g)
            @test MOD.ismetric(h)        == false
            @test MOD.isinnerprod(h)     == false
            @test MOD.attainsnegative(h) == MOD.attainsnegative(g)
            @test MOD.attainszero(h)     == MOD.attainszero(g)
            @test MOD.attainspositive(h) == MOD.attainspositive(g)

            @test MOD.isnonnegative(h) == !MOD.attainsnegative(h)
            @test MOD.ispositive(h)    == (!MOD.attainsnegative(h) && !MOD.attainszero(h))
            @test MOD.isnegative(h)    == (!MOD.attainspositive(h) && !MOD.attainszero(h))

            show(DevNull, h)
        else
            @test_throws ErrorException g ∘ f
        end
    end
end

for h_obj in composite_functions
    @test typeof((h_obj)()) <: CompositeFunction
end

for f_obj in pairwise_functions
    f = (f_obj)()

    if MOD.iscomposable(PolynomialClass(), f)
        @test typeof((f^2).g) <: PolynomialClass
    else
        @test_throws ErrorException f^2
    end

    if MOD.iscomposable(PowerClass(), f)
        @test typeof((f^0.5).g) <: PowerClass
   else
        @test_throws ErrorException f^0.5
    end

    if MOD.iscomposable(ExponentiatedClass(), f)
        @test typeof((exp(f)).g) <: ExponentiatedClass
    else
        @test_throws ErrorException exp(f)
    end

    if MOD.iscomposable(SigmoidClass(), f)
        @test typeof((tanh(f)).g) <: SigmoidClass
    else
        @test_throws ErrorException tanh(f)
    end

end
