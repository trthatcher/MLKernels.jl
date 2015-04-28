using Base.Test

importall MLKernels

#####
# All tests on Float64 only
# Float32 does not have enough precision to calculate finite-difference gradients to sufficient accuracy!
#####

# compares numerical derivative (f(p+ϵ)-f(p-ϵ))/2ϵ with symbolic derivative to ensure correctness
function checkderiv(f, fprime, p, i; eps=1e-3)
    pplus = copy(p); pplus[i] += eps
    pminus = copy(p); pminus[i] -= eps
    return (fprime(p,i) - (f(pplus)-f(pminus))/2eps)
end

checkderiv(f, fprime, p; eps=1e-3) = eltype(p)[checkderiv(f, fprime, p, i; eps=eps) for i=1:length(p)]

checkderivvec(f, fprime, x; eps=1e-3) = abs(checkderiv(f, (p,i)->fprime(p)[i], x; eps=eps))

function test_deriv_dxdy(k, x, y, epsilon)
    @test all(checkderivvec(p->kernel(k,p,y), p->dkernel_dx(k,p,y), x) .< epsilon)
    @test all(checkderivvec(p->kernel(k,x,p), p->dkernel_dy(k,x,p), y) .< epsilon)

    for i=1:length(x), j=1:length(y)
        @test abs(checkderiv(
            p -> dkernel_dx(k,x,p)[i],
            (p,j_) -> d2kernel_dxdy(k,x,p)[i,j_],
            y, j)) < epsilon
        @test abs(checkderiv(
            p -> dkernel_dy(k,p,y)[j],
            (p,i_) -> d2kernel_dxdy(k,p,y)[i_,j],
            x, i)) < epsilon
    end
end

function test_deriv_dp(kconstructor, param, derivs, x, y, epsilon)
    @assert length(param) == length(derivs)
    k = kconstructor(param)
    for (i, deriv) in enumerate(derivs)
        @test abs(checkderiv(
            p -> kernel(kconstructor(p), x, y),
            (p,i_) -> dkernel_dp(kconstructor(p), i_, x, y),
            param, i)) < epsilon
        @test dkernel_dp(k, deriv, x, y) == dkernel_dp(k, i, x, y)
    end
    @test dkernel_dp(k, :undefined, x, y) == zero(eltype(x))
    @test_throws Exception dkernel_dp(k, 0, x, y)
    @test_throws Exception dkernel_dp(k, length(derivs)+1, x, y)
end

print("- Testing EuclideanDistanceKernel derivatives ... ")
for T in (Float64,)
    x = T[1, 2, 7, 3]
    y = T[5, 2, 1, 6]

    @test all(checkderivvec(p->MLKernels.norm2(p,y), p->MLKernels.dnorm2_dx(p,y), x) .< 1e-10)
    @test all(checkderivvec(p->MLKernels.norm2(x,p), p->MLKernels.dnorm2_dy(x,p), y) .< 1e-10)

    for (k, param, derivs) in (
            (GaussianKernel, T[3.0], (:sigma,)),)
        test_deriv_dxdy(k(param...), x, y, 1e-9)
        test_deriv_dp(p->k(p...), param, derivs, x, y, 1e-8)
    end
end
println("Done")

print("- Testing simple composite kernel derivatives ... ")
for T in (Float64,)
    x = T[1, 2, 7, 3]
    y = T[5, 2, 1, 6]

    kproductconstructor(param) = param[1] * GaussianKernel(param[2]) * GaussianKernel(param[3])
    ksumconstructor(param) = param[1]*GaussianKernel(param[2]) + param[3]*GaussianKernel(param[4])
    kscaledconstructor(param) = param[1]*GaussianKernel(param[2])

    for (kconst, param, derivs) in (
            (kproductconstructor, T[3.2, 1.5, 1.8], (:a, symbol("k1.sigma"), symbol("k2.sigma"))),
            (ksumconstructor, T[0.4, 3.2, 1.5, 1.8], (:a1, symbol("k1.sigma"), :a2, symbol("k2.sigma"))),
            (kscaledconstructor, T[0.4, 3.2], (:a, symbol("k.sigma"))))
        test_deriv_dxdy(kconst(param), x, y, 1e-9)
        test_deriv_dp(kconst, param, derivs, x, y, 1e-7)
    end
end
println("Done")
