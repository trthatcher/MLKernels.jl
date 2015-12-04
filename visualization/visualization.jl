function generate_surface{T<:AbstractFloat}(
        lower::Tuple{T,T},
        upper::Tuple{T,T},
        dims::Tuple{Int64,Int64},
    )
    lower[1] < upper[1] || error("Lower must be less than upper on all dimensions")
    lower[2] < upper[2] || error("Lower must be less than upper on all dimensions")
    X = Array(T, prod(dims), 2)
    y_step = (upper[2] - lower[2])/(dims[1]-1)
    x_step = (upper[1] - lower[1])/(dims[2]-1)
    for j = 1:dims[2], i = 1:dims[1]
        idx = (j-1)*dims[1] + i
        X[idx, 1] = lower[1] + (j-1)*x_step
        X[idx, 2] = upper[2] - (i-1)*y_step
    end
    X
end

# Performs KPCA for 3 dimensions
function kpca_project{T<:Base.LinAlg.BlasReal}(K::Matrix{T})
    Λ, V = LAPACK.syev!('V', 'U', centerkernelmatrix!(copy(K)))
    W = V[:,end:-1:end-2]
    μ = mean(K,2)
    Z = (K .- μ)* W
    Z = Z .- minimum(Z,1)  # Translate to (0,0,0)
    scale!(Z, 1 ./ vec(maximum(Z,1)))  # scale to fit in unit square
end

function coordinate_matrices{T<:AbstractFloat}(W::Matrix{T}, dims)
    size(W,2) == 3 || error("Matrix must be n by 3")
    (reshape(W[:,1], (n_y, n_x)), reshape(W[:,2], (n_y, n_x)), reshape(W[:,3], (n_y, n_x)))
end

using MLKernels, PyPlot

n_x = 25
n_y = 25

dims = (n_y, n_x)

# Gaussian Kernel
W = kpca_project(kernelmatrix(GaussianKernel(), generate_surface((-2.0,-2.0), (2.0,2.0), dims)))
X, Y, Z = coordinate_matrices(W, dims)
PyPlot.figure("Gaussian Kernel" )
PyPlot.plot_wireframe(X, Y, Z)

# Laplacian Kernel
W = kpca_project(kernelmatrix(LaplacianKernel(), generate_surface((-2.0,-2.0), (2.0,2.0), dims)))
X, Y, Z = coordinate_matrices(W, dims)
PyPlot.figure("Laplacian Kernel")
PyPlot.plot_wireframe(X, Y, Z)

# Exponential Kernel with Sine-Squared Kernel
W = kpca_project(kernelmatrix(ExponentialKernel(SineSquaredKernel()), generate_surface((-π/2,-π/2),(π/2,π/2), dims)))
X, Y, Z = coordinate_matrices(W, dims)
PyPlot.figure("Exponential Kernel composed with a Sine-Squared Kernel")
PyPlot.plot_wireframe(X, Y, Z)

# Exponential Kernel with Chi-Squared Kernel
W = kpca_project(kernelmatrix(ExponentialKernel(ChiSquaredKernel()), generate_surface((0.0,0.0), (1.0,1.0), dims)))
X, Y, Z = coordinate_matrices(W, dims)
PyPlot.figure("Exponential Kernel composed with a Chi-Squared Kernel")
PyPlot.plot_wireframe(X, Y, Z)

# Rational-Quadratic Kernel with SquaredDistance Kernel
W = kpca_project(kernelmatrix(RationalQuadraticKernel(SquaredDistanceKernel(), 1.0, 0.1, 1.0),generate_surface((-2.0,-2.0), (2.0,2.0), dims)))
X, Y, Z = coordinate_matrices(W, dims)
PyPlot.figure("Rational-Quadratic Kernel")
PyPlot.plot_wireframe(X, Y, Z)

# Matern Kernel
W = kpca_project(kernelmatrix(MaternKernel(),generate_surface((-2.0,-2.0), (2.0,2.0), dims)))
X, Y, Z = coordinate_matrices(W, dims)
PyPlot.figure("Exponential Kernel composed with a Chi-Squared Kernel")
PyPlot.plot_wireframe(X, Y, Z)

# Exponentiated Kernel
W = kpca_project(kernelmatrix(ExponentiatedKernel(),generate_surface((-1.25,-1.25), (1.25,1.25), dims)))
X, Y, Z = coordinate_matrices(W, dims)
PyPlot.figure("Exponentiated Scalar Product Kernel")
PyPlot.plot_wireframe(X, Y, Z)

# Polynomial Kernel
W = kpca_project(kernelmatrix(PolynomialKernel(),generate_surface((-1.25,-1.25), (1.25,1.25), dims)))
X, Y, Z = coordinate_matrices(W, dims)
PyPlot.figure("Polynomial Kernel")
PyPlot.plot_wireframe(X, Y, Z)
