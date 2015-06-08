using Base.Test

importall MLKernels

function basenystrom(kernel::Kernel, X::Matrix, xs::Vector)
    C = kernelmatrix(kernel, X, X[xs,:])
    D = C[xs,:]
    SVD = svdfact(D)
    DVC = diagm(1./sqrt(SVD[:S])) * SVD[:Vt] * C'
    MLKernels.syml(BLAS.syrk('U', 'T', 1, DVC))
end

print("- Testing kernel matrix approximation ... ")
k = ExponentialKernel()
X = rand(5,3)
@test_approx_eq nystrom(k, X, [1,3,5]) basenystrom(k, X, [1,3,5])
@test_approx_eq nystrom(k, X, [1:5]) kernelmatrix(k, X)
println("Done")
