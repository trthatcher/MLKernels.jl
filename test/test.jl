type Test1{T<:Real}
    alpha::T
    gamma::T
end

test{T<:Real}(k::Test1{T}, z::T) = exp(k.alpha * z ^ k.gamma)

type Test2{T<:Real}
    alpha::T
    beta::T
    gamma::T
end

test{T<:Real}(k::Test2{T}, z::T) = (1 + k.alpha * z ^ k.gamma)^(-k.beta)

n = 50000000

using MLKernels

k2 = RationalClass(2.0, 1.5, 0.5)

X = rand(n);

CASE = MLKernels.checkcase(k2)
t = @elapsed @inbounds for i = 1:n
    MLKernels.phi(k2, CASE, X[i])
end

println(t)

X = rand(n);

k1 = Test2(2.0, 1.5, 0.5)

t = @elapsed @inbounds for i = 1:n
    test(k1, X[i])
end
println(t)
