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
# This completes the surface (or else you're missing a slice of pie)
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

function generate_point_cluster{T<:AbstractFloat}(n, center::Tuple{T,T}, radius::T)
    u = radius * sqrt(rand(n))
    v = 2π*rand(n)
    (u .* hcat(cos(v), sin(v))) .+ [center[1] center[2]]
end

function generate_point_ring{T<:AbstractFloat}(n, center::Tuple{T,T}, band_center::T, width::T)
    u = (rand(n) * width) .+ (band_center - width/2)
    v = rand(n) * 2band_center
    r = Array(T, n)
    for i = 1:n
        r[i] = v[i] < u[i] ? u[i] : 2band_center - u[i]
    end
    θ = 2π*rand(n)
    r .* hcat(cos(θ), sin(θ)) .+ [center[1] center[2]]
end


# Performs a form of KPCA for 3 dimensions
function kpca_3d{T<:Base.LinAlg.BlasReal}(K::Matrix{T})
    Λ, V = LAPACK.syev!('V', 'U', centerkernelmatrix!(copy(K)))
    W = V[:,end:-1:end-2]  # eigenvalues are in ascending order
end


using MLKernels, PyPlot

#===================================================================================================
  Kernel Trick Visualization - Linearly Separable Classes
===================================================================================================#

n_points = 40
n_contours = 15
n_sample = 100

U  = generate_circles((0.0,0.0), 1.5, n_points, n_contours)
c1 = generate_point_cluster(n_sample, (0.0, 0.0), 0.5)
c2 = generate_point_ring(n_sample, (0.0, 0.0), 1.15, 0.3)

PyPlot.figure("Original")
PyPlot.scatter3D(vec(c1[:,1]), vec(c1[:,2]), c="r")
PyPlot.scatter3D(vec(c2[:,1]), vec(c2[:,2]), c="m")

κ = GaussianKernel()
K = kernelmatrix(κ, U)
W = kpca_3d(K)

X, Y, Z = coordinate_matrices(K*W, n_points, n_contours)

PyPlot.figure("Wireframe")
PyPlot.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

κ_c1 = kernelmatrix(κ, c1, U) * W
PyPlot.scatter3D(vec(κ_c1[:,1]), vec(κ_c1[:,2]), vec(κ_c1[:,3]), c="r")

κ_c2 = kernelmatrix(κ, c2, U) * W
PyPlot.scatter3D(vec(κ_c2[:,1]), vec(κ_c2[:,2]), vec(κ_c2[:,3]), c="m")

PyPlot.figure("SeparatingHyperplane")

κ_c1 = kernelmatrix(κ, c1, U) * W
PyPlot.scatter3D(vec(κ_c1[:,1]), vec(κ_c1[:,2]), vec(κ_c1[:,3]), c="r")

κ_c2 = kernelmatrix(κ, c2, U) * W
PyPlot.scatter3D(vec(κ_c2[:,1]), vec(κ_c2[:,2]), vec(κ_c2[:,3]), c="m")

PyPlot.plot_surface([-7.0 7; -7 7], [7.0 7; -7 -7], 3.5ones(2,2), 
                     rstride=1, cstride=1, alpha=0.25)


#===================================================================================================
  Kernel Trick Visualization 
===================================================================================================#

for (κ, title, n_points, n_contours, center, radius) in (
        (SineSquaredKernel(), "Sine-Squared Kernel", 40, 15, (0.0, 0.0), 0.5),
        (ChiSquaredKernel(), "Chi-Squared Kernel", 40, 15, (0.5, 0.5), 0.5),
        (GaussianKernel(), "Gaussian Kernel", 40, 15, (0.0, 0.0), 1.0),
        (LaplacianKernel(), "Laplacian Kernel", 40, 15, (0.0, 0.0), 1.0),
        (PeriodicKernel(), "Periodic Kernel", 40, 15, (0.0, 0.0), 0.5),
        (RationalQuadraticKernel(), "Rational-Quadratic Kernel", 40, 15, (0.0, 0.0), 1.0),
        (MaternKernel(), "Matern Kernel", 40, 15, (0.0, 0.0), 1.0),
        (PolynomialKernel(), "Polynomial Kernel", 40, 15, (0.0, 0.0), 1.0),
        (SigmoidKernel(), "Sigmoid Kernel", 40, 15, (0.0, 0.0), 1.0)
    )
    U = generate_circles(center, radius, n_points, n_contours)
    W = kpca_3d((isnegdef(κ) ? -1 : 1) * kernelmatrix(κ, U))
    X, Y, Z = coordinate_matrices(W, n_points, n_contours)
    PyPlot.figure(title)
    PyPlot.plot_wireframe(X, Y, Z)
end
