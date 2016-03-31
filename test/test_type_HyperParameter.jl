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
        Bu = Bound(one(T),  ustrict)

        I = Interval(Bl, Bu)

        @test get(I.lower) == Bl
        @test get(I.upper) == Bu

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
        B = Bound(zero(T), strict)

        I = LowerBound(B)
        @test isnull(I.upper)
        @test get(I.lower) == B

        I = LowerBound(zero(T), strict ? :strict : :nonstrict)
        @test isnull(I.upper)
        @test get(I.lower) == B

        I = UpperBound(B)
        @test isnull(I.lower)
        @test get(I.upper) == B

        I = UpperBound(zero(T), strict ? :strict : :nonstrict)
        @test isnull(I.lower)
        @test get(I.upper) == B
    end
end
