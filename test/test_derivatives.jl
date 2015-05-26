using Base.Test

importall MLKernels

#####
# All tests on Float64 only
# Float32 does not have enough precision to calculate finite-difference gradients to sufficient accuracy!
#####

# compares numerical derivative (f(p+ϵ)-f(p-ϵ))/2ϵ with symbolic derivative to ensure correctness
function checkderiv(f, fprime, p, i; eps=1e-6)
    pplus = copy(p); pplus[i] += eps
    pminus = copy(p); pminus[i] -= eps
    delta = (fprime(p,i) - (f(pplus)-f(pminus))/2eps)
    #return @show(delta)
    return (delta)
end

checkderiv(f, fprime, p::Number; eps=1e-6) = (fprime(p) - (f(p+eps)-f(p-eps))/2eps)

checkderiv(f, fprime, p::Vector; eps=1e-6) = eltype(p)[checkderiv(f, fprime, p, i; eps=eps) for i=1:length(p)]

checkderivvec(f, fprime, x; eps=1e-6) = abs(checkderiv(f, (p,i)->fprime(p)[i], x; eps=eps))

function test_kappa_dz(k, z, epsilon)
    print("dz ")
    @test_approx_eq_eps checkderiv(p->MLKernels.kappa(k,p), p->MLKernels.kappa_dz(k,p), z) zero(z) epsilon

    if !isa(k, SeparableKernel)
        print("dz2 ")
        @test_approx_eq_eps checkderiv(p->MLKernels.kappa_dz(k,p), p->MLKernels.kappa_dz2(k,p), z) zero(z) epsilon
    end
end

function test_kernel_dxdy(k::Kernel, x::Real, y::Real, epsilon)
    print("scalar ")
    @test_approx_eq_eps checkderiv(p->kernel(k,p,y), p->kernel_dx(k,p,y), x) zero(x) epsilon
    @test_approx_eq_eps checkderiv(p->kernel(k,x,p), p->kernel_dy(k,x,p), y) zero(y) epsilon
    @test_approx_eq_eps checkderiv(p->kernel_dx(k,x,p), p->kernel_dxdy(k,x,p), y) zero(x) epsilon
    @test_approx_eq_eps checkderiv(p->kernel_dy(k,p,y), p->kernel_dxdy(k,p,y), x) zero(y) epsilon
end

function test_kernel_dxdy(k::Kernel, x::Vector, y::Vector, epsilon)
    print("dx ")
    @test_approx_eq_eps checkderivvec(p->kernel(k,p,y), p->kernel_dx(k,p,y), x) zeros(x) epsilon
    print("dy ")
    @test_approx_eq_eps checkderivvec(p->kernel(k,x,p), p->kernel_dy(k,x,p), y) zeros(y) epsilon

    if isa(k, ARD) && isa(k.k, PeriodicKernel)
        warn("!!! NOT TESTED: kernel_dxdy too low precision for ARD{PeriodicKernel} !!!")
        return
    end

    print("dxdy ")
    for i=1:length(x), j=1:length(y)
        @test_approx_eq_eps checkderiv(
            p -> kernel_dx(k,x,p)[i],
            (p,j_) -> kernel_dxdy(k,x,p)[i,j_],
            y, j) zero(eltype(x)) epsilon
        @test_approx_eq_eps checkderiv(
            p -> kernel_dy(k,p,y)[j],
            (p,i_) -> kernel_dxdy(k,p,y)[i_,j],
            x, i) zero(eltype(y)) epsilon
    end

    if isa(k, ARD)
        warn("kernelmatrix_dxdy not implemented for ARD")
        return
    end
    if isa(k, ScaledKernel)
        warn("kernelmatrix_dxdy not implemented for ScaledKernel")
        return
    end
    if isa(k, CompositeKernel)
        warn("kernelmatrix_dxdy not implemented for CompositeKernel")
        return
    end
    print("matrix ")
    @test_approx_eq kernel_dx(k,x,y) kernelmatrix_dx(k, x'', y'', 'T')
    @test_approx_eq kernel_dx(k,x,y) kernelmatrix_dx(k, x', y')
    @test_approx_eq kernel_dy(k,x,y) kernelmatrix_dy(k, x'', y'', 'T')
    @test_approx_eq kernel_dy(k,x,y) kernelmatrix_dy(k, x', y')
    @test_approx_eq kernel_dxdy(k,x,y) kernelmatrix_dxdy(k, x'', y'', 'T')
    @test_approx_eq kernel_dxdy(k,x,y) kernelmatrix_dxdy(k, x', y')
end

function test_kappa_dp(ktype, param, derivs, z, epsilon)
    @assert length(param) == length(derivs)
    print("d")
    k = ktype(param...)
    @test MLKernels.kappa_dp(k, :undefined, z) == zero(eltype(z))
    for (i, deriv) in enumerate(derivs)
        print(":$deriv")
        @test_approx_eq_eps checkderiv(
            p -> MLKernels.kappa(ktype(p...), z),
            (p,i_) -> MLKernels.kappa_dp(ktype(p...), deriv, z),
            param, i) zero(eltype(z)) epsilon
        @eval kappa_deriv = MLKernels.$(symbol("kappa_d$(deriv)"))
        @test MLKernels.kappa_dp(k, deriv, z) == kappa_deriv(k, z)
    end
    print(" ")
end

function test_kernel_dp(ktype, param, derivs, x, y, epsilon)
    @assert length(param) == length(derivs)
    print("d")
    k = ktype(param...)
    for (i, deriv) in enumerate(derivs)
        print(":$deriv")
        @test_approx_eq_eps checkderiv(
            p -> kernel(ktype(p...), x, y),
            (p,i_) -> kernel_dp(ktype(p...), i_, x, y),
            param, i) 0.0 epsilon
        @test kernel_dp(k, deriv, x, y) == kernel_dp(k, i, x, y)
    end
    @test kernel_dp(k, :undefined, x, y) == 0.0
    @test_throws Exception kernel_dp(k, 0, x, y)
    @test_throws Exception kernel_dp(k, length(derivs)+1, x, y)
    print(" ")
end

println("- Testing vector function derivatives:")
for T in (Float64,)
    x = T[1, 2, 7, 3]
    y = T[5, 2, 1, 6]
    w = T[4, 1, 6, 0]

    for s_fun in (:sqdist, :scprod)
        print("- Testing $(s_fun) ... ")
        fun = @eval(MLKernels.$s_fun)
        for d in (:x, :y, :w)
            @eval $(symbol("fun_d$(d)")) = MLKernels.$(symbol("$(s_fun)_d$(d)"))
        end

        @test_approx_eq_eps checkderivvec(p->fun(p,y), p->fun_dx(p,y), x) zeros(x) 1e-7
        @test_approx_eq_eps checkderivvec(p->fun(x,p), p->fun_dy(x,p), y) zeros(y) 1e-7

        @test_approx_eq_eps checkderivvec(p->fun(p,y,w), p->fun_dx(p,y,w), x) zeros(x) 1e-7
        @test_approx_eq_eps checkderivvec(p->fun(x,p,w), p->fun_dy(x,p,w), y) zeros(y) 1e-7

        @test_approx_eq_eps checkderivvec(p->fun(x,y,p), p->fun_dw(x,y,p), w) zeros(w) 1e-7
        println("Done")
    end
end

println("- Testing standard kernel derivatives")
for T in (Float64,)
    x = T[1, 2, -7, 3]
    y = T[5, 2, 1, 6]
    z = convert(T, 1.5)

    for (ktype, param, derivs) in (
            (SquaredExponentialKernel, T[0.3], (:alpha,)),
            (GammaExponentialKernel, T[1.3, 0.45], (:alpha, :gamma)),
            (InverseQuadraticKernel, T[1.3], (:alpha,)),
            (RationalQuadraticKernel, T[1.3, 2.1], (:alpha, :beta)),
            (GammaRationalQuadraticKernel, T[1.3, 2.1, 0.6], (:alpha, :beta, :gamma)),
            (GammaPowerKernel, T[0.8], (:gamma,)),
            (LogKernel, T[1, 0.5], (:alpha, :gamma)),
            (PeriodicKernel, T[1.1, 1.3], (:p, :ell)),
            (LinearKernel, T[1.2], (:c,)),
            (PolynomialKernel, T[1.1, 1.3, 2.2], (:alpha, :c, :d)),
            (SigmoidKernel, T[1.1, 1.3], (:alpha, :c)),
            (MercerSigmoidKernel, T[1.1, 1.3], (:d, :b)),
        )
        print("    - Testing $(ktype) ... ")
        k = ktype(param...)
        test_kappa_dz(k, z, 5e-7)
        test_kernel_dxdy(k, x[1], y[1], 1e-7)
        test_kernel_dxdy(k, x, y, 1e-5)
        test_kappa_dp(ktype, param, derivs, z, 6e-5)
        test_kernel_dp(ktype, param, derivs, x, y, 6e-5)
        test_kernel_dp(ktype, param, derivs, x[1], y[1], 1e-7)
        println("Done")
    end
end

println("- Testing ARD kernel:")
for T in (Float64,)
    x = T[1, 2, -7, 3]
    y = T[5, 2, 1, 6]
    w = T[0.0, 0.5, 1.5, 1.0]

    for (ktype, param, derivs) in (
            (SquaredExponentialKernel, T[0.3], (:alpha,)),
            (GammaExponentialKernel, T[1.3, 0.45], (:alpha, :gamma)),
            (InverseQuadraticKernel, T[1.3], (:alpha,)),
            (RationalQuadraticKernel, T[1.3, 2.1], (:alpha, :beta)),
            (GammaPowerKernel, T[0.8], (:gamma,)),
            #(LogKernel, T[1], (:d,)),
            (PeriodicKernel, T[1.1, 1.3], (:p, :ell)),
            (LinearKernel, T[1.2], (:c,)),
            (PolynomialKernel, T[1.1, 1.3, 2.2], (:alpha, :c, :d)),
            (SigmoidKernel, T[1.1, 1.3], (:alpha, :c)),
        )
        print("    - Testing ARD{$(ktype)} ... ")
        test_kernel_dxdy(ARD(ktype(param...), w), x, y, 2e-6)
        test_kernel_dxdy(ARD(ktype(param...), length(x)), x, y, 1e-6)
        #test_kernel_dp(ktype, param, derivs, x, y, 6e-5)
        println("Done")
    end
end

println("- Testing simple composite kernel derivatives:")
for T in (Float64,)
    x = T[1, 2, 7, 3]
    y = T[5, 2, 1, 6]

    kproductconstructor(param...) = param[1] * SquaredExponentialKernel(param[2]) * SquaredExponentialKernel(param[3])
    ksumconstructor(param...) = param[1]*SquaredExponentialKernel(param[2]) + param[3]*SquaredExponentialKernel(param[4])
    kscaledconstructor(param...) = param[1]*SquaredExponentialKernel(param[2])

    for (kconst, param, derivs) in (
            (kproductconstructor, T[3.2, 1.5, 1.8], (:a, symbol("k1.alpha"), symbol("k2.alpha"))),
            (ksumconstructor, T[0.4, 3.2, 1.5, 1.8], (:a1, symbol("k1.alpha"), :a2, symbol("k2.alpha"))),
            (kscaledconstructor, T[0.4, 3.2], (:a, symbol("k.alpha"))))
        k = kconst(param...)
        print("    - Testing $k ... ")
        test_kernel_dxdy(k, x, y, 1e-9)
        test_kernel_dp(kconst, param, derivs, x, y, 1e-7)
        println("Done")
    end
end
