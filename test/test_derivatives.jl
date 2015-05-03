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
    @test all(checkderivvec(p->kernel(k,p,y), p->kernel_dx(k,p,y), x) .< epsilon)
    @test all(checkderivvec(p->kernel(k,x,p), p->kernel_dy(k,x,p), y) .< epsilon)

    for i=1:length(x), j=1:length(y)
        @test abs(checkderiv(
            p -> kernel_dx(k,x,p)[i],
            (p,j_) -> kernel_dxdy(k,x,p)[i,j_],
            y, j)) < epsilon
        @test abs(checkderiv(
            p -> kernel_dy(k,p,y)[j],
            (p,i_) -> kernel_dxdy(k,p,y)[i_,j],
            x, i)) < epsilon
    end
end

function test_deriv_dp(kconstructor, param, derivs, x, y, epsilon)
    @assert length(param) == length(derivs)
    k = kconstructor(param)
    for (i, deriv) in enumerate(derivs)
        @test abs(checkderiv(
            p -> kernel(kconstructor(p), x, y),
            (p,i_) -> kernel_dp(kconstructor(p), i_, x, y),
            param, i)) < epsilon
        @test kernel_dp(k, deriv, x, y) == kernel_dp(k, i, x, y)
    end
    @test kernel_dp(k, :undefined, x, y) == zero(eltype(x))
    @test_throws Exception kernel_dp(k, 0, x, y)
    @test_throws Exception kernel_dp(k, length(derivs)+1, x, y)
end

print("- Testing vector function derivatives ... ")
for T in (Float64,)
    x = T[1, 2, 7, 3]
    y = T[5, 2, 1, 6]
    w = T[4, 1, 6, 0]

    for s_fun in (:sqdist, :scprod)
        fun = @eval(MLKernels.$s_fun)
        for d in (:x, :y, :w)
            @eval $(symbol("fun_d$(d)")) = MLKernels.$(symbol("$(s_fun)_d$(d)"))
        end

        @test all(checkderivvec(p->fun(p,y), p->fun_dx(p,y), x) .< 1e-10)
        @test all(checkderivvec(p->fun(x,p), p->fun_dy(x,p), y) .< 1e-10)

        @test all(checkderivvec(p->fun(p,y,w), p->fun_dx(p,y,w), x) .< 1e-10)
        @test all(checkderivvec(p->fun(x,p,w), p->fun_dy(x,p,w), y) .< 1e-10)

        @test all(checkderivvec(p->fun(x,y,p), p->fun_dw(x,y,p), w) .< 1e-10)
    end
end
println("Done")

print("- Testing SquaredDistanceKernel derivatives ... ")
for T in (Float64,)
    x = T[1, 2, 7, 3]
    y = T[5, 2, 1, 6]

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
