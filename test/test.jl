type Test{T<:Real}
    alpha::T
    gamma::T
end

test{T<:Real}(k::Test{T}, z::T) = exp(k.alpha * z ^ k.gamma)

t = 0.0;
n = 10000000
X = rand(n);

k1 = Test(2.0, 0.7)

t = @elapsed @inbounds for i = 1:n
    test(k1, X[i])
end


println(t)

using MLKernels

k2 = ExponentialClass(2.0, 0.7)

X = rand(n);

t = @elapsed @inbounds for i = 1:n
    MLKernels.phi(k2, X[i])
end

print(t)
