using Base.Test

importall MLKernels

# compares numerical derivative (f(p+ϵ)-f(p-ϵ))/2ϵ with symbolic derivative to ensure correctness
function checkderiv(f, fprime, p, i; eps=1e-3)
    pplus = copy(p); pplus[i] += eps
    pminus = copy(p); pminus[i] -= eps
    return fprime(p,i) - (f(pplus)-f(pminus))/2eps
end

checkderiv(f, fprime, p; eps=1e-3) = eltype(p)[checkderiv(f, fprime, p, i; eps=eps) for i=1:length(p)]

checkderivvec(f, fprime, x; eps=1e-3) = abs(checkderiv(f, (p,i)->fprime(p)[i], x; eps=eps))

print("- Testing EuclideanDistanceKernel derivatives ... ")
for T in (Float64,)
    x = T[1, 2, 7, 3]
    y = T[5, 2, 1, 6]

    @test all(checkderivvec(p->MLKernels.euclidean_distance(p,y), p->MLKernels.deuclidean_distance_dx(p,y), x) .< 1e-10)
    @test all(checkderivvec(p->MLKernels.euclidean_distance(x,p), p->MLKernels.deuclidean_distance_dy(x,p), y) .< 1e-10)

    param = T[3.0]

    k = GaussianKernel(param...)
    @test all(checkderivvec(p->kernel(k,p,y), p->dkernel_dx(k,p,y), x) .< 1e-9)
    @test all(checkderivvec(p->kernel(k,x,p), p->dkernel_dy(k,x,p), y) .< 1e-9)

    for j=1:4, i=1:4
        @test abs(checkderiv(
            p->dkernel_dx(k,x,p)[j],
            (p,i_)->d2kernel_dxdy(k,x,p)[j,i_],
            y, i)) < 1e-9
        @test abs(checkderiv(
            p->dkernel_dy(k,p,y)[i],
            (p,j_)->d2kernel_dxdy(k,p,y)[j_,i],
            x, j)) < 1e-9
    end

    @test abs(checkderiv(
        p->kernel(GaussianKernel(p...),x,y),
        (p,i)->dkernel_dp(GaussianKernel(p...),:sigma,x,y),
        param, 1)) < 1e-8
    @test dkernel_dp(k, :sigma, x, y) == dkernel_dp(k, 1, x, y)
    @test dkernel_dp(k, :undefined, x, y) == zero(T)
    @test_throws ArgumentError dkernel_dp(k, 0)
    @test_throws ArgumentError dkernel_dp(k, 2)
    println("Done")
end
