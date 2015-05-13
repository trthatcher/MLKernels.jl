using Base.Test

importall MLKernels

#####
# All tests on Float64 only
# Float32 does not have enough precision to calculate finite-difference gradients to sufficient accuracy!
#####

# compares numerical derivative (f(p+ϵ)-f(p-ϵ))/2ϵ with symbolic derivative to ensure correctness
function checkderiv(f, fprime, p, i; eps=1e-4)
    pplus = copy(p); pplus[i] += eps
    pminus = copy(p); pminus[i] -= eps
    delta = (fprime(p,i) - (f(pplus)-f(pminus))/2eps)
    #return @show(delta)
    return (delta)
end

checkderiv(f, fprime, p::Number; eps=1e-4) = (fprime(p) - (f(p+eps)-f(p-eps))/2eps)

checkderiv(f, fprime, p::Vector; eps=1e-4) = eltype(p)[checkderiv(f, fprime, p, i; eps=eps) for i=1:length(p)]

checkderivvec(f, fprime, x; eps=1e-4) = abs(checkderiv(f, (p,i)->fprime(p)[i], x; eps=eps))

function test_kappa_dz(k, z, epsilon)
    print("dz ")
    @test abs(checkderiv(p->MLKernels.kappa(k,p), p->MLKernels.kappa_dz(k,p), z)) < epsilon

    if !isa(k, SeparableKernel)
        print("dz2 ")
        @test abs(checkderiv(p->MLKernels.kappa_dz(k,p), p->MLKernels.kappa_dz2(k,p), z)) < epsilon
    end
end

function test_kernel_dxdy(k, x, y, epsilon)
    print("dx ")
    @test all(checkderivvec(p->kernel(k,p,y), p->kernel_dx(k,p,y), x) .< epsilon)
    print("dy ")
    @test all(checkderivvec(p->kernel(k,x,p), p->kernel_dy(k,x,p), y) .< epsilon)

    isa(k, ARD) && return #XXX TODO not implemented yet

    print("dxdy ")
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

function test_kappa_dp(kconstructor, param, derivs, z, epsilon)
    @assert length(param) == length(derivs)
    print("d")
    k = kconstructor(param...)
    for (i, deriv) in enumerate(derivs)
        print(":$deriv")
        @test abs(checkderiv(
            p -> MLKernels.kappa(kconstructor(p...), z),
            (p,i_) -> MLKernels.kappa_dp(kconstructor(p...), deriv, z),
            param, i)) < epsilon
    end
    @test MLKernels.kappa_dp(k, :undefined, z) == zero(eltype(z))
    print(" ")
end

function test_kernel_dp(kconstructor, param, derivs, x, y, epsilon)
    @assert length(param) == length(derivs)
    print("d")
    k = kconstructor(param...)
    for (i, deriv) in enumerate(derivs)
        print(":$deriv")
        @test abs(checkderiv(
            p -> kernel(kconstructor(p...), x, y),
            (p,i_) -> kernel_dp(kconstructor(p...), i_, x, y),
            param, i)) < epsilon
        @test kernel_dp(k, deriv, x, y) == kernel_dp(k, i, x, y)
    end
    @test kernel_dp(k, :undefined, x, y) == zero(eltype(x))
    @test_throws Exception kernel_dp(k, 0, x, y)
    @test_throws Exception kernel_dp(k, length(derivs)+1, x, y)
    print(" ")
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

        @test all(checkderivvec(p->fun(p,y), p->fun_dx(p,y), x) .< 1e-8)
        @test all(checkderivvec(p->fun(x,p), p->fun_dy(x,p), y) .< 1e-8)

        @test all(checkderivvec(p->fun(p,y,w), p->fun_dx(p,y,w), x) .< 1e-8)
        @test all(checkderivvec(p->fun(x,p,w), p->fun_dy(x,p,w), y) .< 1e-8)

        @test all(checkderivvec(p->fun(x,y,p), p->fun_dw(x,y,p), w) .< 1e-8)
    end
end
println("Done")

println("- Testing standard kernel derivatives")
for T in (Float64,)
    x = T[1, 2, -7, 3]
    y = T[5, 2, 1, 6]
    z = convert(T, 1.5)

    for (k, param, derivs) in (
            (GaussianKernel, T[3.0], (:sigma,)),
            (LaplacianKernel, T[1.3], (:sigma,)),
            (RationalQuadraticKernel, T[1.3], (:c,)),
            (MultiQuadraticKernel, T[1.3], (:c,)),
            (InverseMultiQuadraticKernel, T[1.3], (:c,)),
            (PowerKernel, T[2], (:d,)),
            (LogKernel, T[1], (:d,)),
            (LinearKernel, T[1.2], (:c,)),
            (PolynomialKernel, T[1.1, 1.3, 2.2], (:alpha, :c, :d)),
            (SigmoidKernel, T[1.1, 1.3], (:alpha, :c)),
            (MercerSigmoidKernel, T[1.1, 1.3], (:d, :b)),
        )
        print("    - Testing $(k) ... ")
        test_kappa_dz(k(param...), z, 1e-7)
        test_kernel_dxdy(k(param...), x, y, 1e-7)
        test_kappa_dp(k, param, derivs, z, 6e-5)
        test_kernel_dp(k, param, derivs, x, y, 6e-5)
        println("Done")
    end
end

println("- Testing ARD kernel:")
for T in (Float64,)
    x = T[1, 2, -7, 3]
    y = T[5, 2, 1, 6]
    w = T[0.0, 0.5, 1.5, 1.0]

    for (k, param, derivs) in (
            (GaussianKernel, T[3.0], (:sigma,)),
            (LaplacianKernel, T[1.3], (:sigma,)),
            (RationalQuadraticKernel, T[1.3], (:c,)),
            (MultiQuadraticKernel, T[1.3], (:c,)),
            (InverseMultiQuadraticKernel, T[1.3], (:c,)),
            (PowerKernel, T[2], (:d,)),
            (LogKernel, T[1], (:d,)),
            (LinearKernel, T[1.2], (:c,)),
            (PolynomialKernel, T[1.1, 1.3, 2.2], (:alpha, :c, :d)),
            (SigmoidKernel, T[1.1, 1.3], (:alpha, :c)),
        )
        print("    - Testing ARD{$(k)} ... ")
        test_kernel_dxdy(ARD(k(param...), w), x, y, 5e-7)
        test_kernel_dxdy(ARD(k(param...), length(x)), x, y, 5e-7)
        #test_kernel_dp(k, param, derivs, x, y, 6e-5)
        println("Done")
    end
end

println("- Testing simple composite kernel derivatives:")
for T in (Float64,)
    x = T[1, 2, 7, 3]
    y = T[5, 2, 1, 6]

    kproductconstructor(param...) = param[1] * GaussianKernel(param[2]) * GaussianKernel(param[3])
    ksumconstructor(param...) = param[1]*GaussianKernel(param[2]) + param[3]*GaussianKernel(param[4])
    kscaledconstructor(param...) = param[1]*GaussianKernel(param[2])

    for (kconst, param, derivs) in (
            (kproductconstructor, T[3.2, 1.5, 1.8], (:a, symbol("k1.sigma"), symbol("k2.sigma"))),
            (ksumconstructor, T[0.4, 3.2, 1.5, 1.8], (:a1, symbol("k1.sigma"), :a2, symbol("k2.sigma"))),
            (kscaledconstructor, T[0.4, 3.2], (:a, symbol("k.sigma"))))
        k = kconst(param...)
        print("    - Testing $k ... ")
        test_kernel_dxdy(k, x, y, 1e-9)
        test_kernel_dp(kconst, param, derivs, x, y, 1e-7)
        println("Done")
    end
end
