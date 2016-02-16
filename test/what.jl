type Test1{T<:Real}
    a::T
end

immutable Wrapper{T<:Real}
    a::T
end

type Test2{T<:Real}
    w::Wrapper{T}
end

test{T<:Real}(x::Test1{T}) = x.a
test{T<:Real}(x::Test2{T}) = x.w.a

T1 = Test1{Float64}(2.0)
T2 = Test2{Float64}(Wrapper{Float64}(2.0))


