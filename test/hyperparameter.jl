info("Testing ", MOD.OpenBound)
for T in (FloatingPointTypes..., IntegerTypes...)
    if T <: Integer
        @test_throws ErrorException MOD.OpenBound(one(T))
    else
        if T <: AbstractFloat
            @test_throws ErrorException MOD.OpenBound(convert(T,  NaN))
            @test_throws ErrorException MOD.OpenBound(convert(T,  Inf))
            @test_throws ErrorException MOD.OpenBound(convert(T, -Inf))
        end

        for x in (1,2)
            B = MOD.OpenBound(convert(T,x))

            @test B.value  == x
            @test eltype(B) == T
        end

        for U in FloatingPointTypes
            B = MOD.OpenBound(one(T))

            B_u = convert(MOD.OpenBound{U}, B)
            @test B_u.value == one(U)
            @test eltype(B_u) == U
        end

        B = MOD.OpenBound(zero(T))

        @test checkvalue(-one(T), B) == true
        @test checkvalue(zero(T), B) == false
        @test checkvalue(one(T),  B) == false

        @test checkvalue(B, -one(T)) == false
        @test checkvalue(B, zero(T)) == false
        @test checkvalue(B, one(T))  == true
    end
end

info("Testing ", MOD.ClosedBound)
for T in (FloatingPointTypes..., IntegerTypes...)
    if T <: AbstractFloat
        @test_throws ErrorException MOD.ClosedBound(convert(T,  NaN))
        @test_throws ErrorException MOD.ClosedBound(convert(T,  Inf))
        @test_throws ErrorException MOD.ClosedBound(convert(T, -Inf))
    end

    for x in (1,2)
        B = MOD.ClosedBound(convert(T,x))

        @test B.value  == x
        @test eltype(B) == T
    end

    for U in (FloatingPointTypes..., IntegerTypes...)
        B = MOD.ClosedBound(one(T))

        B_u = convert(MOD.ClosedBound{U}, B)
        @test B_u.value == one(U)
        @test eltype(B_u) == U
    end

    B = MOD.ClosedBound(zero(T))

    @test checkvalue(-one(T), B) == true
    @test checkvalue(zero(T), B) == true
    @test checkvalue(one(T),  B) == false

    @test checkvalue(B, -one(T)) == false
    @test checkvalue(B, zero(T)) == true
    @test checkvalue(B, one(T))  == true
end

info("Testing ", MOD.NullBound)
for T in (FloatingPointTypes..., IntegerTypes...)
    @test eltype(NullBound(T)) == T

    for U in (FloatingPointTypes..., IntegerTypes...)
        B = MOD.NullBound(T)

        B_u = convert(MOD.NullBound{U}, B)
        @test eltype(B_u) == U
    end

end

info("Testing ", MOD.Interval)
for T in (FloatingPointTypes..., IntegerTypes...)
    for lisopen in (true, false), uisopen in (true, false)
        Bl = Bound(zero(T), lisopen)
        Br = Bound(convert(T,2),  uisopen)

        I = Interval(Bl, Br)

        @test eltype(I) == T

        @test get(I.left) == Bl
        @test get(I.right) == Br

        @test MOD.checkinterval(I, -one(T)) == false
        @test MOD.checkinterval(I, convert(T,3)) == false

        @test MOD.checkinterval(I, one(T))

        if lisopen
            @test MOD.checkinterval(I, zero(T)) == false
        else
            @test MOD.checkinterval(I, zero(T)) == true
        end

        if uisopen
            @test MOD.checkinterval(I, convert(T,2)) == false
        else
            @test MOD.checkinterval(I, convert(T,2)) == true
        end

        for lnull in (true, false), unull in (true, false)
            I = Interval(lnull ? Nullable{Bound{T}}() : Nullable(Bl),
                         unull ? Nullable{Bound{T}}() : Nullable(Br))
            @test lnull ? isnull(I.left) : get(I.left) == Bl
            @test unull ? isnull(I.right) : get(I.right) == Br

            if T in FloatingPointTypes
                I2 = convert(Interval{Float32}, I)
                @test lnull ? isnull(I2.left) : get(I2.left) == convert(Bound{Float32}, Bl)
                @test unull ? isnull(I2.right) : get(I2.right) == convert(Bound{Float32}, Br)
                I2 = convert(Interval{Float64}, I)
                @test lnull ? isnull(I2.left) : get(I2.left) == convert(Bound{Float64}, Bl)
                @test unull ? isnull(I2.right) : get(I2.right) == convert(Bound{Float64}, Br)
            end

            # Test that output does not create error
            show(DevNull, I)
        end

        Br = Bound(zero(T), uisopen)

        if lisopen || uisopen
            @test_throws ErrorException Interval(Bl, Br)
        else
            I = Interval(Bl, Br)
            @test get(I.left) == Bl
            @test get(I.right) == Br
        end
    end
    for isopen in (true, false)
        B = Bound(convert(T,2), isopen)

        I = leftbounded(B)
        @test isnull(I.right)
        @test get(I.left) == B
        @test MOD.checkinterval(I, one(T)) == false
        @test MOD.checkinterval(I, convert(T,3)) == true
        @test MOD.checkinterval(I, convert(T,2)) == (isopen ? false : true)

        I = leftbounded(convert(T,2), isopen ? :open : :closed)
        @test isnull(I.right)
        @test get(I.left) == B

        @test_throws ErrorException leftbounded(one(T), :test)

        I = rightbounded(B)
        @test isnull(I.left)
        @test get(I.right) == B
        @test MOD.checkinterval(I, one(T)) == true
        @test MOD.checkinterval(I, convert(T,3)) == false
        @test MOD.checkinterval(I, convert(T,2)) == (isopen ? false : true)

        I = rightbounded(convert(T,2), isopen ? :open : :closed)
        @test isnull(I.left)
        @test get(I.right) == B

        @test_throws ErrorException rightbounded(zero(T), :test)

        I = unbounded(T)
        @test MOD.checkinterval(I, -one(T)) == true
        @test MOD.checkinterval(I, zero(T)) == true
        @test MOD.checkinterval(I,  one(T)) == true
    end
end

info("Testing ", MOD.HyperParameter)
for T in (FloatingPointTypes..., IntegerTypes...)
    I = rightbounded(one(T), :open)

    P = HyperParameter(zero(T), I)
    @test P.value == zero(T)

    @test_throws ErrorException HyperParameter(one(T), I)

    show(DevNull, HyperParameter(zero(T), I))

    P = HyperParameter(convert(T,2), unbounded(T))
end
