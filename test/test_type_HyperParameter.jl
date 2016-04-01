info("Testing ", Bound)
for T in (FloatingPointTypes..., IntegerTypes...)
    for strict in (true, false)
        B = Bound(one(T), strict)
        @test B.value == one(T)
        @test B.is_strict == strict
        @test B == Bound(one(T), strict ? :strict : :nonstrict)

        if T in FloatingPointTypes
            @test eltype(convert(Bound{Float32}, B)) == Float32
            @test eltype(convert(Bound{Float64}, B)) == Float64
        else
            @test eltype(convert(Bound{Int32}, B)) == Int32
            @test eltype(convert(Bound{Int64}, B)) == Int64
        end

        @test_throws ErrorException Bound(one(T), :test)
    end
end

info("Testing ", Interval)
for T in (FloatingPointTypes..., IntegerTypes...)
    for lstrict in (true, false), ustrict in (true, false)
        Bl = Bound(zero(T), lstrict)
        Bu = Bound(convert(T,2),  ustrict)

        I = Interval(Bl, Bu)

        @test eltype(I) == T

        @test get(I.lower) == Bl
        @test get(I.upper) == Bu

        @test MOD.checkbounds(I, -one(T)) == false
        @test MOD.checkbounds(I, convert(T,3)) == false

        @test MOD.checkbounds(I, one(T))

        if lstrict
            @test MOD.checkbounds(I, zero(T)) == false
        else
            @test MOD.checkbounds(I, zero(T)) == true
        end

        if ustrict
            @test MOD.checkbounds(I, convert(T,2)) == false
        else
            @test MOD.checkbounds(I, convert(T,2)) == true
        end

        for lnull in (true, false), unull in (true, false)
            I = Interval(lnull ? Nullable{Bound{T}}() : Nullable(Bl),
                         unull ? Nullable{Bound{T}}() : Nullable(Bu))
            @test lnull ? isnull(I.lower) : get(I.lower) == Bl
            @test unull ? isnull(I.upper) : get(I.upper) == Bu

            if T in FloatingPointTypes
                I2 = convert(Interval{Float32}, I)
                @test lnull ? isnull(I2.lower) : get(I2.lower) == convert(Bound{Float32}, Bl)
                @test unull ? isnull(I2.upper) : get(I2.upper) == convert(Bound{Float32}, Bu)
                I2 = convert(Interval{Float64}, I)
                @test lnull ? isnull(I2.lower) : get(I2.lower) == convert(Bound{Float64}, Bl)
                @test unull ? isnull(I2.upper) : get(I2.upper) == convert(Bound{Float64}, Bu)
            end

            # Test that output does not create error
            show(DevNull, I)
        end

        Bu = Bound(zero(T), ustrict)

        if lstrict || ustrict
            @test_throws ErrorException Interval(Bl, Bu)
        else
            I = Interval(Bl, Bu)
            @test get(I.lower) == Bl
            @test get(I.upper) == Bu
        end
    end
    for strict in (true, false)
        B = Bound(convert(T,2), strict)

        I = LowerBound(B)
        @test isnull(I.upper)
        @test get(I.lower) == B
        @test MOD.checkbounds(I, one(T)) == false
        @test MOD.checkbounds(I, convert(T,3)) == true
        @test MOD.checkbounds(I, convert(T,2)) == (strict ? false : true)

        I = LowerBound(convert(T,2), strict ? :strict : :nonstrict)
        @test isnull(I.upper)
        @test get(I.lower) == B

        @test_throws ErrorException LowerBound(one(T), :test)

        I = UpperBound(B)
        @test isnull(I.lower)
        @test get(I.upper) == B
        @test MOD.checkbounds(I, one(T)) == true
        @test MOD.checkbounds(I, convert(T,3)) == false
        @test MOD.checkbounds(I, convert(T,2)) == (strict ? false : true)

        I = UpperBound(convert(T,2), strict ? :strict : :nonstrict)
        @test isnull(I.lower)
        @test get(I.upper) == B

        @test_throws ErrorException UpperBound(zero(T), :test)
    end
end

info("Testing ", Variable)
for T in (FloatingPointTypes..., IntegerTypes...)

    v = Variable(one(T))
    @test v.value   == one(T)
    @test v.isfixed == false

    @test Variable(one(T), true).value   == one(T)
    @test Variable(one(T), true).isfixed == false

    @test Fixed(one(T)).value   == one(T)
    @test Fixed(one(T)).isfixed == true

    if T in FloatingPointTypes
        @test eltype(convert(Variable{Float32}, v)) == Float32
        @test eltype(convert(Variable{Float64}, v)) == Float64
    else
        @test eltype(convert(Variable{Int32}, v)) == Int32
        @test eltype(convert(Variable{Int64}, v)) == Int64
    end
end
