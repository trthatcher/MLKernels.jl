using Base.Test

importall MLKernels

#####
# All tests on Float64 only
# Float32 does not have enough precision to calculate finite-difference gradients to sufficient accuracy!
#####

# compares numerical derivative (f(p+ϵ)-f(p-ϵ))/2ϵ with symbolic derivative to ensure correctness

function checkderiv(f, fprime, p::Vector, idx; doprint=false)
    doprint && println("epsilon   df-f'    f'/df")

    fp = fprime(p, idx)

    deltas = zeros(eltype(p), 9)
    epsilon = 1e-2
    for i=1:length(deltas)
        pplus = copy(p); pplus[idx] += epsilon
        pminus = copy(p); pminus[idx] -= epsilon
        df = (f(pplus) - f(pminus))/2epsilon

        doprint && @printf("%10s %20.15f %20.15f\n", "10^(-$(i+1))", df-fp, fp/df)
        deltas[i] = abs(df-fp)

        epsilon /= 10
    end

    minimum(deltas)
end

function checkderiv(f, fprime, p::Number; doprint=false)
    checkderiv(x->f(x[1]), (x,i)->fprime(x[1]), [p], 1; doprint=doprint)
end

checkderiv(f, fprime, p::Vector) = eltype(p)[checkderiv(f, fprime, p, i) for i=1:length(p)]

checkderivvec(f, fprime, x) = abs(checkderiv(f, (p,i)->fprime(p)[i], x))

function test_kappa_dz(k, z, epsilon)
    print("dz ")
    @test_approx_eq_eps checkderiv(p->MLKernels.kappa(k,p), p->MLKernels.kappa_dz(k,p), z) zero(z) epsilon

    if !isa(k, SeparableKernel) # SeparableKernels only need to define kappa_dz
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
        deriv == nothing && continue
        print(":$deriv")
        if isa(deriv, Tuple)
            @test_throws deriv[2] MLKernels.kappa_dp(k, deriv[1], z)
        else
            @test_approx_eq_eps checkderiv(
                p -> MLKernels.kappa(ktype(p...), z),
                (p,i_) -> MLKernels.kappa_dp(ktype(p...), deriv, z),
                param, i) zero(eltype(z)) epsilon
            @eval kappa_deriv = MLKernels.$(symbol("kappa_d$(deriv)"))
            @test MLKernels.kappa_dp(k, deriv, z) == kappa_deriv(k, z)
        end
    end
    print(" ")
end

function test_kernel_dp(ktype, param, derivs, x, y, epsilon)
    @assert length(param) == length(derivs)
    k = ktype(param...)

    print("d")
    for (i, deriv) in enumerate(derivs)
        deriv == nothing && continue
        print(":$deriv")
        if isa(deriv, Tuple)
            @test_throws deriv[2] MLKernels.kernel_dp(k, deriv[1], x, y)
            @test_throws deriv[2] MLKernels.kernel_dp(k, i, x, y)
        else
            @test_approx_eq_eps checkderiv(
                p -> kernel(ktype(p...), x, y),
                (p,i_) -> kernel_dp(ktype(p...), i_, x, y),
                param, i) 0.0 epsilon
            @test kernel_dp(k, deriv, x, y) == kernel_dp(k, i, x, y)
            if isa(x, Array)
                @test kernelmatrix_dp(k, deriv, x', y') == kernelmatrix_dp(k, i, x', y')
                @test_approx_eq kernel_dp(k, deriv, x, y) kernelmatrix_dp(k, i, x'', y'', 'T')
                @test_approx_eq kernel_dp(k, deriv, x, y) kernelmatrix_dp(k, i, x', y')
                @test_approx_eq kernel_dp(k, deriv, x, x) kernelmatrix_dp(k, i, x')
                @test_approx_eq kernel_dp(k, deriv, x, x) kernelmatrix_dp(k, i, x'', 'T')
            end
        end
    end
    @test kernel_dp(k, :undefined, x, y) == 0.0
    @test_throws Union(BoundsError,ArgumentError) kernel_dp(k, 0, x, y)
    @test_throws Union(BoundsError,ArgumentError) kernel_dp(k, length(derivs)+1, x, y)
    print(" ")
end

function test_kernel_ard_dp(ktype, weights, param, derivs, x, y, epsilon)
    @assert length(param) == length(derivs)
    k = ARD(ktype(param...), weights)

    print("d")
    for (i, deriv) in enumerate(derivs)
        deriv == nothing && continue
        print(":$deriv")
        if isa(deriv, Tuple)
            @test_throws deriv[2] MLKernels.kernel_dp(k, deriv[1], x, y)
            @test_throws deriv[2] MLKernels.kernel_dp(k, i+1, x, y)
        else
            @test_approx_eq_eps checkderiv(
                p -> kernel(ARD(ktype(p...), weights), x, y),
                (p,i_) -> kernel_dp(ARD(ktype(p...), weights), i_+1, x, y),
                param, i) 0.0 epsilon
            @test kernel_dp(k, deriv, x, y) == kernel_dp(k, i+1, x, y)
        end
    end
    @test_approx_eq_eps checkderiv(
        w -> kernel(ARD(ktype(param...), w), x, y),
        (w,i_) -> kernel_dp(ARD(ktype(param...), w), :weights, x, y)[i_],
        weights) zeros(weights) epsilon
    @test kernel_dp(k, :weights, x, y) == kernel_dp(k, 1, x, y)
    @test kernel_dp(k, :undefined, x, y) == 0.0
    @test_throws Union(BoundsError,ArgumentError) kernel_dp(k, 0, x, y)
    @test_throws Union(BoundsError,ArgumentError) kernel_dp(k, length(derivs)+2, x, y)
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

        @test_approx_eq_eps checkderivvec(p->fun(p,y), p->fun_dx(p,y), x) zeros(x) 1e-9
        @test_approx_eq_eps checkderivvec(p->fun(x,p), p->fun_dy(x,p), y) zeros(y) 1e-9

        @test_approx_eq_eps checkderivvec(p->fun(p,y,w), p->fun_dx(p,y,w), x) zeros(x) 1e-9
        @test_approx_eq_eps checkderivvec(p->fun(x,p,w), p->fun_dy(x,p,w), y) zeros(y) 1e-9

        @test_approx_eq_eps checkderivvec(p->fun(x,y,p), p->fun_dw(x,y,p), w) zeros(w) 1e-9
        println("Done")
    end
end

println("- Testing standard kernel derivatives")
for T in (Float64,)
    x = T[1, 2, -7, 3]
    y = T[3, 2, 1, 6]
    z = convert(T, 1.5)

    for (ktype, param, derivs) in (
            (ExponentialKernel, T[1.3, 0.45], (:alpha, :gamma)),
            (ExponentialKernel, T[1.3, 1], (:alpha, nothing)),
            (RationalQuadraticKernel, T[1.3, 2.1, 0.6], (:alpha, :beta, :gamma)),
            (RationalQuadraticKernel, T[1.3, 1, 0.6], (:alpha, :beta, :gamma)),
            (RationalQuadraticKernel, T[1.3, 2.1, 1], (:alpha, :beta, nothing)),
            (RationalQuadraticKernel, T[1.3, 1, 1], (:alpha, :beta, nothing)),
            (PowerKernel, T[0.8], (:gamma,)),
            (PowerKernel, T[1], (nothing,)),
            (LogKernel, T[1.1, 0.5], (:alpha, :gamma)),
            (LogKernel, T[1.1, 1], (:alpha, nothing)),
            (PolynomialKernel, T[1.1, 1.3, 2], (:alpha, :c, (:d,Exception))),
            (PolynomialKernel, T[1.1, 1.3, 1], (:alpha, :c, (:d,Exception))),
            (SigmoidKernel, T[1.1, 1.3], (:alpha, :c)),
            (MercerSigmoidKernel, T[1.1, 1.3], (:d, :b)),
            (PeriodicKernel, T[1.1, 1.3], (:period, :ell)),
        )
        print("    - Testing $(ktype) ... ")
        k = ktype(param...)
        if ktype <: PeriodicKernel
            epsilon = 1e-4
        elseif ktype <: PolynomialKernel
            epsilon = 5e-7
        elseif ktype <: PowerKernel
            epsilon = 2e-8
        else
            epsilon = 1e-9
        end
        test_kappa_dz(k, z, epsilon)
        test_kernel_dxdy(k, x[1], y[1], epsilon)
        test_kernel_dxdy(k, x, y, epsilon)
        test_kappa_dp(ktype, param, derivs, z, epsilon)
        test_kernel_dp(ktype, param, derivs, x, y, epsilon)
        test_kernel_dp(ktype, param, derivs, x[1], y[1], epsilon)
        println("Done")
    end
end

println("- Testing ARD kernel:")
for T in (Float64,)
    x = T[1, 2, -7, 3]
    y = T[5, 2, 1, 6]
    w = T[0.0, 0.5, 1.5, 1.0]
    w2 = T[0.1, 0.5, 1.5, 1.0] # finite difference only possible for non-zero weight

    for (ktype, param, derivs) in (
            (ExponentialKernel, T[1.3, 0.45], (:alpha, :gamma)),
            (RationalQuadraticKernel, T[1.3, 2.1, 0.6], (:alpha, :beta, :gamma)),
            (PowerKernel, T[0.8], (:gamma,)),
            #(LogKernel, T[1], (:d,)),
            #(PeriodicKernel, T[1.1, 1.3], (:period, :ell)),
            (PolynomialKernel, T[1.1, 1.3, 2], (:alpha, :c, (:d,Exception))),
            (SigmoidKernel, T[1.1, 1.3], (:alpha, :c)),
        )
        print("    - Testing ARD{$(ktype)} ... ")
        test_kernel_dxdy(ARD(ktype(param...), w), x, y, 1e-7)
        test_kernel_dxdy(ARD(ktype(param...), length(x)), x, y, 1e-7)
        test_kernel_ard_dp(ktype, w2, param, derivs, x, y, 1e-7)
        println("Done")
    end
end

println("- Testing simple composite kernel derivatives:")
for T in (Float64,)
    x = T[1, 2, 5, 3]
    y = T[2, 2, 1, 6]

    kproductconstructor(param...) = param[1] * ExponentialKernel(param[2], param[3]) * ExponentialKernel(param[4], param[5])
    ksumconstructor(param...) = ExponentialKernel(param[1], param[2]) + ExponentialKernel(param[3], param[4])
    kscaledconstructor(param...) = param[1]*ExponentialKernel(param[2], param[3])

    for (kconst, param, derivs) in (
            (kproductconstructor, T[3.2, 1.5, 0.9, 1.8, 0.9], (:a, symbol("k[1].alpha"), symbol("k[1].gamma"), symbol("k[2].alpha"), symbol("k[2].gamma"))),
            (ksumconstructor, T[3.2, 0.9, 1.8, 0.9], (symbol("k[1].alpha"), symbol("k[1].gamma"), symbol("k[2].alpha"), symbol("k[2].gamma"))),
            (kscaledconstructor, T[0.4, 3.2, 0.9], (:a, symbol("k[1].alpha"), symbol("k[1].gamma")))
        )
        k = kconst(param...)
        print("    - Testing $k ... ")
        test_kernel_dxdy(k, x, y, 1e-9)
        test_kernel_dxdy(k, x[1], y[1], 1e-9)
        test_kernel_dp(kconst, param, derivs, x, y, 1e-9)
        test_kernel_dp(kconst, param, derivs, x[1], y[1], 1e-9)
        println("Done")
    end

    K1 = ExponentialKernel(one(T))
    K2 = RationalQuadraticKernel(one(T))
    K3 = PolynomialKernel(one(T))
    K4 = SigmoidKernel(one(T))

    for K1K2 in (K1*K2, K1+K2, K1*K2*K3*K4, K1+K2+K3+K4, K1*K2+K3*K4,
                 K1*(K2+K3)*K4, (K1+K2)*(K3+K4), K1*K3+K1*K4+K2*K3+K2*K4)
        print("    - Testing $K1K2 ... ")
        test_kernel_dxdy(K1K2, x, y, 1e-7)
        test_kernel_dxdy(K1K2, x[1], y[1], 1e-7)
        println("Done")
    end
      
end
