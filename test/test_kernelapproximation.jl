#=
function basenystrom(kernel::Kernel, X::Matrix, xs::Vector)
    C = kernelmatrix(kernel, X, X[xs,:])
    D = C[xs,:]
    SVD = svdfact(D)
    S = 1 ./ sqrt(SVD[:S])
    S[abs(SVD[:S]) .< 1e-10] = 0 # remove infinities
    DVC = diagm(S) * SVD[:Vt] * C'
    MLKernels.syml(BLAS.syrk('U', 'T', 1, DVC))
end
=#

info("Testing ", nystrom)
k = GaussianKernel()
X = rand(5,3)
@test_approx_eq nystrom(k, X, collect(1:5)) kernelmatrix(k, X)
