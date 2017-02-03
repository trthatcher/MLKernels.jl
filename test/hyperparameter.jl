info("Testing ", MOD.Bound)
for T in (FloatingPointTypes..., IntegerTypes...)
    for isopen in (true, false)
        B = Bound(one(T), isopen)
        @test B.value == one(T)
        @test B.isopen == isopen
        @test B == Bound(one(T), isopen ? :open : :closed)

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

info("Testing ", MOD.Interval)
for T in (FloatingPointTypes..., IntegerTypes...)
    for lisopen in (true, false), uisopen in (true, false)
        Bl = Bound(zero(T), lisopen)
        Br = Bound(convert(T,2),  uisopen)

        I = Interval(Bl, Br)

        @test eltype(I) == T

        @test get(I.left) == Bl
        @test get(I.right) == Br

        @test MOD.checkbounds(I, -one(T)) == false
        @test MOD.checkbounds(I, convert(T,3)) == false

        @test MOD.checkbounds(I, one(T))

        if lisopen
            @test MOD.checkbounds(I, zero(T)) == false
        else
            @test MOD.checkbounds(I, zero(T)) == true
        end

        if uisopen
            @test MOD.checkbounds(I, convert(T,2)) == false
        else
            @test MOD.checkbounds(I, convert(T,2)) == true
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
        @test MOD.checkbounds(I, one(T)) == false
        @test MOD.checkbounds(I, convert(T,3)) == true
        @test MOD.checkbounds(I, convert(T,2)) == (isopen ? false : true)

        I = leftbounded(convert(T,2), isopen ? :open : :closed)
        @test isnull(I.right)
        @test get(I.left) == B

        @test_throws ErrorException leftbounded(one(T), :test)

        I = rightbounded(B)
        @test isnull(I.left)
        @test get(I.right) == B
        @test MOD.checkbounds(I, one(T)) == true
        @test MOD.checkbounds(I, convert(T,3)) == false
        @test MOD.checkbounds(I, convert(T,2)) == (isopen ? false : true)

        I = rightbounded(convert(T,2), isopen ? :open : :closed)
        @test isnull(I.left)
        @test get(I.right) == B

        @test_throws ErrorException rightbounded(zero(T), :test)

        I = unbounded(T)
        @test MOD.checkbounds(I, -one(T)) == true
        @test MOD.checkbounds(I, zero(T)) == true
        @test MOD.checkbounds(I,  one(T)) == true
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
