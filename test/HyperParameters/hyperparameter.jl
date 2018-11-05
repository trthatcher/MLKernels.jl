@testset "Testing MLK.OpenBound" begin
    for T in (FloatingPointTypes..., IntegerTypes...)
        if T <: Integer
            @test_throws ErrorException MLK.OpenBound(one(T))
        else
            if T <: AbstractFloat
                @test_throws ErrorException MLK.OpenBound(convert(T,  NaN))
                @test_throws ErrorException MLK.OpenBound(convert(T,  Inf))
                @test_throws ErrorException MLK.OpenBound(convert(T, -Inf))
            end

            for x in (1,2)
                B = MLK.OpenBound(convert(T,x))

                @test B.value  == x
                @test eltype(B) == T
            end

            for U in FloatingPointTypes
                B = MLK.OpenBound(one(T))

                B_u = convert(MLK.OpenBound{U}, B)
                @test B_u.value == one(U)
                @test eltype(B_u) == U
            end

            B = MLK.OpenBound(zero(T))

            @test checkvalue(-one(T), B) == true
            @test checkvalue(zero(T), B) == false
            @test checkvalue(one(T),  B) == false

            @test checkvalue(B, -one(T)) == false
            @test checkvalue(B, zero(T)) == false
            @test checkvalue(B, one(T))  == true

            @test string(B) == string("OpenBound(", zero(T), ")")
            @test show(devnull, B) == nothing
        end
    end
end

@testset "Testing MLK.ClosedBound" begin
    for T in (FloatingPointTypes..., IntegerTypes...)
        if T <: AbstractFloat
            @test_throws ErrorException MLK.ClosedBound(convert(T,  NaN))
            @test_throws ErrorException MLK.ClosedBound(convert(T,  Inf))
            @test_throws ErrorException MLK.ClosedBound(convert(T, -Inf))
        end

        for x in (1,2)
            B = MLK.ClosedBound(convert(T,x))

            @test B.value  == x
            @test eltype(B) == T
        end

        for U in (FloatingPointTypes..., IntegerTypes...)
            B = MLK.ClosedBound(one(T))

            B_u = convert(MLK.ClosedBound{U}, B)
            @test B_u.value == one(U)
            @test eltype(B_u) == U
        end

        B = MLK.ClosedBound(zero(T))

        @test checkvalue(-one(T), B) == true
        @test checkvalue(zero(T), B) == true
        @test checkvalue(one(T),  B) == false

        @test checkvalue(B, -one(T)) == false
        @test checkvalue(B, zero(T)) == true
        @test checkvalue(B, one(T))  == true

        @test string(B) == string("ClosedBound(", zero(T), ")")
        @test show(devnull, B) == nothing
    end
end

@testset "Testing MLK.NullBound" begin
    for T in (FloatingPointTypes..., IntegerTypes...)
        @test eltype(NullBound(T)) == T

        for U in (FloatingPointTypes..., IntegerTypes...)
            B = MLK.NullBound(T)

            B_u = convert(MLK.NullBound{U}, B)
            @test eltype(B_u) == U
        end

        B = MLK.NullBound(T)

        @test checkvalue(-one(T), B) == true
        @test checkvalue(zero(T), B) == true
        @test checkvalue(one(T),  B) == true

        @test checkvalue(B, -one(T)) == true
        @test checkvalue(B, zero(T)) == true
        @test checkvalue(B, one(T))  == true

        @test string(B) == string("NullBound(", T, ")")
        @test show(devnull, B) == nothing
    end
end

@testset "Testing MLK.Interval" begin
    for T in FloatingPointTypes
        @test_throws ErrorException MLK.Interval(MLK.ClosedBound(one(T)), MLK.OpenBound(one(T)))
        @test_throws ErrorException MLK.Interval(MLK.OpenBound(one(T)),   MLK.ClosedBound(one(T)))
        @test_throws ErrorException MLK.Interval(MLK.OpenBound(one(T)),   MLK.OpenBound(one(T)))

        a = MLK.ClosedBound(one(T))
        b = MLK.ClosedBound(one(T))
        I = MLK.Interval(a,b)
        @test I.a == a
        @test I.b == b
        @test eltype(I) == T

        I = MLK.interval(nothing, nothing)
        @test I.a == NullBound(Float64)
        @test I.b == NullBound(Float64)
        @test eltype(I) == Float64

        for a in (NullBound(T), ClosedBound(-one(T)), OpenBound(-one(T)))
            for b in (NullBound(T), ClosedBound(one(T)), OpenBound(one(T)))
                I = MLK.Interval(a, b)
                @test I.a == a
                @test I.b == b
                @test eltype(I) == T

                if typeof(a) <: NullBound
                    if typeof(b) <: NullBound
                        @test I == MLK.interval(T)
                        @test string(I) == string("interval(", T, ")")
                    else
                        @test I == MLK.interval(nothing, b)
                        @test string(I) == string("interval(nothing,", string(b), ")")
                    end
                else
                    if typeof(b) <: NullBound
                        @test I == MLK.interval(a,nothing)
                        @test string(I) == string("interval(", string(a), ",nothing)")
                    else
                        @test I == MLK.interval(a, b)
                        @test string(I) == string("interval(", string(a), ",", string(b), ")")
                    end
                end

                @test MLK.checkvalue(I, convert(T,-2)) == (typeof(a) <: NullBound ? true  : false)
                @test MLK.checkvalue(I, -one(T))       == (typeof(a) <: OpenBound ? false : true)
                @test MLK.checkvalue(I, zero(T))       == true
                @test MLK.checkvalue(I,  one(T))       == (typeof(b) <: OpenBound ? false : true)
                @test MLK.checkvalue(I, convert(T,2))  == (typeof(b) <: NullBound ? true  : false)
            end
        end

        a = convert(T,7.6)
        b = convert(T,23)
        c = convert(T,13)

        @test show(devnull, MLK.interval(MLK.ClosedBound(one(T)), MLK.ClosedBound(one(T)))) == nothing
    end
end

@testset "Testing MLK.interval" begin
    @test typeof(MLK.interval(nothing, nothing)) == MLK.Interval{Float64,NullBound{Float64},NullBound{Float64}}
    for T in FloatingPointTypes
        for a in (NullBound(T), ClosedBound(-one(T)), OpenBound(-one(T)))
            for b in (NullBound(T), ClosedBound(one(T)), OpenBound(one(T)))
                null_a = typeof(a) <: NullBound
                null_b = typeof(b) <: NullBound

                I = interval(null_a ? nothing : a, null_b ? nothing : b)

                if null_a && null_b
                    @test typeof(I) == MLK.Interval{Float64,NullBound{Float64},NullBound{Float64}}
                else
                    @test I.a == a
                    @test I.b == b
                    @test eltype(T) == T
                end
            end
        end
    end
end


@testset "Testing MLK.checkvalue" begin
    for T in FloatingPointTypes
        l = convert(T,-9)
        a = convert(T,-3)
        b = convert(T,3)
        u = convert(T,9)
        for A in (NullBound(T), ClosedBound(a), OpenBound(a))
            T_a = typeof(A)
            for B in (NullBound(T), ClosedBound(b), OpenBound(b))
                T_b = typeof(B)
                I = HP.interval(A,B)

                for x in range(l, stop = a, length = 30)
                    if T_a <: NullBound || T_a <: ClosedBound && x == a
                        @test MLK.checkvalue(I,x) == true
                    else
                        @test MLK.checkvalue(I,x) == false
                    end
                end

                for x in range(a, stop = b, length = 30)
                    if x == a
                        @test MLK.checkvalue(I,x) == T_a <: OpenBound ? false : true
                    elseif x == b
                        @test MLK.checkvalue(I,x) == T_b <: OpenBound ? false : true
                    else
                        @test MLK.checkvalue(I,x) == true
                    end
                end

                for x in range(b, stop = u, length = 30)
                    if T_b <: NullBound || T_b <: ClosedBound && x == b
                        @test MLK.checkvalue(I,x) == true
                    else
                        @test MLK.checkvalue(I,x) == false
                    end
                end
            end
        end

    end
end


@testset "Testing MLK.lowerboundtheta" begin
    for T in FloatingPointTypes
        for A in (NullBound(T), ClosedBound(-one(T)), OpenBound(-one(T)))
            T_a = typeof(A)
            for B in (NullBound(T), ClosedBound(one(T)), OpenBound(one(T)))
                T_b = typeof(B)
                I = MLK.Interval(A, B)

                if T_a <: NullBound
                    if T_b <: NullBound
                        @test MLK.lowerboundtheta(I) == convert(T,-Inf)
                    elseif T_b <: ClosedBound
                        @test MLK.lowerboundtheta(I) == convert(T,-Inf)
                    else
                        @test MLK.lowerboundtheta(I) == convert(T,-Inf)
                    end
                elseif T_a <: ClosedBound
                    if T_b <: NullBound
                        @test MLK.lowerboundtheta(I) == I.a.value
                    elseif T_b <: ClosedBound
                        @test MLK.lowerboundtheta(I) == I.a.value
                    else
                        @test MLK.lowerboundtheta(I) == convert(T,-Inf)
                    end
                else
                    if T_b <: NullBound
                        @test MLK.lowerboundtheta(I) == convert(T,-Inf)
                    elseif T_b <: ClosedBound
                        @test MLK.lowerboundtheta(I) == convert(T,-Inf)
                    else
                        @test MLK.lowerboundtheta(I) == convert(T,-Inf)
                    end
                end
            end
        end
    end
end

@testset "Testing MLK.upperboundtheta" begin
    for T in FloatingPointTypes
        a = convert(T,-5)
        b = convert(T,5)
        for A in (NullBound(T), ClosedBound(a), OpenBound(a))
            T_a = typeof(A)
            for B in (NullBound(T), ClosedBound(b), OpenBound(b))
                T_b = typeof(B)
                I = MLK.Interval(A, B)

                if T_a <: NullBound
                    if T_b <: NullBound
                        @test MLK.upperboundtheta(I) == convert(T,Inf)
                    elseif T_b <: ClosedBound
                        @test MLK.upperboundtheta(I) == b
                    else
                        @test MLK.upperboundtheta(I) == convert(T,Inf)
                    end
                elseif T_a <: ClosedBound
                    if T_b <: NullBound
                        @test MLK.upperboundtheta(I) == convert(T,Inf)
                    elseif T_b <: ClosedBound
                        @test MLK.upperboundtheta(I) == b
                    else
                        @test MLK.upperboundtheta(I) == log(b - a)
                    end
                else
                    if T_b <: NullBound
                        @test MLK.upperboundtheta(I) == convert(T,Inf)
                    elseif T_b <: ClosedBound
                        @test MLK.upperboundtheta(I) == log(b - a)
                    else
                        @test MLK.upperboundtheta(I) == convert(T,Inf)
                    end
                end
            end
        end
    end
end


@testset "Testing HP.theta" begin
    for T in FloatingPointTypes
        l = convert(T,-9)
        a = convert(T,-3)
        b = convert(T,3)
        u = convert(T,9)
        for A in (NullBound(T), ClosedBound(a), OpenBound(a))
            T_a = typeof(A)
            for B in (NullBound(T), ClosedBound(b), OpenBound(b))
                T_b = typeof(B)
                I = MLK.Interval(A, B)

                for x in range(l, stop = u, length = 60)
                    if T_a <: NullBound
                        if T_b <: NullBound
                            @test HP.theta(I,x) == x
                        elseif T_b <: ClosedBound
                            if x <= b
                                @test HP.theta(I,x) == x
                            else
                                @test_throws DomainError HP.theta(I,x)
                            end
                        else
                            if x < b
                                @test HP.theta(I,x) == log(b-x)
                            else
                                @test_throws DomainError HP.theta(I,x)
                            end
                        end
                    elseif T_a <: ClosedBound
                        if T_b <: NullBound
                            if a <= x
                                @test HP.theta(I,x) == x
                            else
                                @test_throws DomainError HP.theta(I,x)
                            end
                        elseif T_b <: ClosedBound
                            if a <= x <= b
                                @test HP.theta(I,x) == x
                            else
                                @test_throws DomainError HP.theta(I,x)
                            end
                        else
                            if a <= x < b
                                @test HP.theta(I,x) == log(b-x)
                            else
                                @test_throws DomainError HP.theta(I,x)
                            end
                        end
                    else
                        if T_b <: NullBound
                            if a < x
                                @test HP.theta(I,x) == log(x-a)
                            else
                                @test_throws DomainError HP.theta(I,x)
                            end
                        elseif T_b <: ClosedBound
                            if a < x <= b
                                @test HP.theta(I,x) == log(x-a)
                            else
                                @test_throws DomainError HP.theta(I,x)
                            end
                        else
                            if a < x < b
                                @test HP.theta(I,x) == log(x-a) - log(b-x)
                            else
                                @test_throws DomainError HP.theta(I,x)
                            end
                        end
                    end
                end
            end
        end
    end
end


@testset "Testing HP.eta" begin
    for T in FloatingPointTypes
        a = -one(T)
        b = one(T)
        for A in (NullBound(T), ClosedBound(a), OpenBound(a))
            for B in (NullBound(T), ClosedBound(b), OpenBound(b))
                I = HP.interval(A,B)
                l_theta = max(HP.lowerboundtheta(I),convert(T,-10))
                u_theta = min(HP.upperboundtheta(I),convert(T,10))

                for x in range(convert(T, -10), stop = convert(T, 10), length = 201)
                    if l_theta <= x <= u_theta
                        v = HP.eta(I,x)
                        @test isapprox(HP.theta(I,v), x)
                    else
                        @test_throws DomainError HP.eta(I,x)
                    end
                end
            end
        end
    end
end


@testset "Testing MLK.HyperParameter" begin
    for T in (FloatingPointTypes..., IntegerTypes...)
        I = interval(nothing, ClosedBound(zero(T)))

        @test_throws ErrorException HyperParameter(one(T), I)

        P = HyperParameter(one(T), interval(nothing, ClosedBound(one(T))))
        @test getindex(P.value) == one(T)
        @test eltype(P) == T

        @test getvalue(P) == one(T)

        @test checkvalue(P, convert(T,2)) == false
        @test checkvalue(P, zero(T)) == true

        setvalue!(P, zero(T))
        @test getindex(P.value) == zero(T)
        @test getvalue(P) == zero(T)

        if T in FloatingPointTypes
            for a in (NullBound(T), ClosedBound(-one(T)), OpenBound(-one(T)))
                for b in (NullBound(T), ClosedBound(one(T)), OpenBound(one(T)))
                    I = HP.Interval(a, b)
                    P = HyperParameter(zero(T),I)

                    @test isapprox(HP.gettheta(P), HP.theta(I, zero(T)))

                    HP.settheta!(P, HP.theta(I, convert(T,0.5)))
                    @test isapprox(HP.gettheta(P), HP.theta(I, convert(T,0.5)))
                    @test isapprox(HP.getvalue(P), convert(T,0.5))
                end
            end
        end

        P1 = HyperParameter(zero(T), interval(T))
        P2 = HyperParameter(one(T),  interval(T))
        for op in (isless, ==, +, -, *, /)
            @test op(P1, one(T)) == op(getvalue(P1), one(T))
            @test op(one(T), P1) == op(one(T), getvalue(P1))
            @test op(P1, P2)     == op(getvalue(P1), getvalue(P2))
        end

        @test show(devnull, P) == nothing
    end
end
