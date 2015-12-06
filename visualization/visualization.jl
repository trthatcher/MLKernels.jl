# Generates a series of circles that will form the surface
function generate_circles{T<:AbstractFloat}(
        center::Tuple{T,T},
        radius::T,
        n_points::Int64,   # Points on a contour
        n_contours::Int64  # Number of contours
    )
    n_contours >= 2 || error("NO")
    X = Array(T, n_points*n_contours, 2)
    angle_step = 2π/n_points
    radius_step = radius/(n_contours-1)
    for j = 1:n_contours 
        r = (n_contours-j) * radius_step        
        for i = 1:n_points
            idx = (j-1)*n_points + i
            θ = (i-1)*angle_step
            X[idx, 1] = center[1] + r * cos(θ)
            X[idx, 2] = center[2] + r * sin(θ)
        end
    end
    X
end

# Returns an n_rows+1 by n_cols Array with the last row being a copy of the first row
#   Purpose: Completes the surface or else you're missing a slice of pie
function circular_reshape{T<:AbstractFloat}(A::Array{T}, n_rows::Int64, n_cols::Int64)
    length(A) == n_rows*n_cols || throw(DimensionMismatch("Dimensions do not conform"))
    T[A[(j-1)*n_rows + mod(i-1, n_rows) + 1] for i = 1:n_rows+1, j = 1:n_cols]
end

function coordinate_matrices{T<:AbstractFloat}(W::Matrix{T}, n_points::Int64, n_contours::Int64)
    size(W,2) == 3 || error("Matrix must be n by 3")
    X = circular_reshape(W[:,1], n_points, n_contours)
    Y = circular_reshape(W[:,2], n_points, n_contours)
    Z = circular_reshape(W[:,3], n_points, n_contours)
    (X, Y, Z)
end

function generate_points{T<:AbstractFloat}(n, center::Tuple{T,T}, radius::T)
    u = one(T) .- sqrt(rand(n))
    v = 2π*rand(n)
    radius * hcat(cos(v) .* u, sin(v) .* u) .+ [center[1] center[2]]
end

# Performs a form of KPCA for 3 dimensions
function kpca_3d{T<:Base.LinAlg.BlasReal}(K::Matrix{T})
    Λ, V = LAPACK.syev!('V', 'U', centerkernelmatrix!(copy(K)))
    W = V[:,end:-1:end-2]  # eigenvalues are in ascending order
end


using MLKernels, PyPlot

n_x = 25
n_y = 25

dims = (n_y, n_x)

#=
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

=#

n_points = 50
n_contours = 20

# Gaussian Kernel
U = generate_circles((0.0,0.0), 2.0, n_points, n_contours)
K = kernelmatrix(GaussianKernel(), U)
W = kpca_3d(centerkernelmatrix(K))
X, Y, Z = coordinate_matrices(K*W, n_points, n_contours)
PyPlot.figure("Gaussian Kernel")
tick_params(axis="both", which="both", bottom="off", top="off")
PyPlot.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=PyPlot.ColorMap("Greys"))
C2 = kernelmatrix(GaussianKernel(), generate_points(150, (.5,.5), 1.0), U) * W
PyPlot.scatter3D(vec(C2[:,1]), vec(C2[:,2]), vec(C2[:,3]), c="r")

# Polynomial Kernel
#=
U = generate_circles((0.0,0.0), 1.0, n_points, n_contours)
K = kernelmatrix(PolynomialKernel(), U)
W = kpca_3d(centerkernelmatrix(K))
X, Y, Z = coordinate_matrices(K*W, n_points, n_contours)
PyPlot.figure("Polynomial Kernel" )
PyPlot.plot_surface(X, Y, Z, rstride=1, cstride=1)
=#

#=
W = kpca_project(kernelmatrix(PolynomialKernel(), ))
X, Y, Z = coordinate_matrices(W, n_points, n_contours)
PyPlot.figure("Polynomial Kernel")
PyPlot.plot_surface(X, Y, Z, rstride=1, cstride=1)
=#

