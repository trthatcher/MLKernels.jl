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
    W = V[:,end:-1:end-2] * 10  # eigenvalues are in ascending order
end


using MLKernels, PyPlot

#===================================================================================================
  Kernel Trick Visualization - Linearly Separable Classes
===================================================================================================#

n_c1 = 60
n_c2 = 180

cluster = 0.55
ring    = (0.725, 0.975)  # radius 1 & 2

xy_mid = (cluster^2 + ring[1]^2)/2
z_lim  = ring[2] * cluster * sqrt(2)

# Class Sampling
srand(1234)
c1 = generate_point_cluster(n_c1, (0.0, 0.0), cluster)
c2 = generate_point_ring(n_c2, (0.0, 0.0), sum(ring)/2, ring[2]-ring[1])

# Plot points in original feature space
PyPlot.figure("FeatureSpace")
PyPlot.scatter(vec(c1[:,1]), vec(c1[:,2]), c="r", marker="D")
PyPlot.scatter(vec(c2[:,1]), vec(c2[:,2]), c="m", marker="o")
PyPlot.xlim([-1.05,1.05])
PyPlot.ylim([-1.05,1.05])
PyPlot.title("Original Feature Space")
PyPlot.xlabel("Feature 1") 
PyPlot.ylabel("Feature 2")

# Explicit Feature Map (Polynomial with d = 2, c = 0)
#   (x,y) -> (x^2, y^2, √2*x*y)
ϕ(x,y) = [x^2 y^2 √2*x*y]
c1_ϕ = vcat([ϕ(c1[i,:]...) for i = 1:size(c1,1)]...)
c2_ϕ = vcat([ϕ(c2[i,:]...) for i = 1:size(c2,1)]...)

# Plot points in kernel Hilbert space
PyPlot.figure("HilbertSpace")
PyPlot.scatter3D(vec(c1_ϕ[:,1]), vec(c1_ϕ[:,2]), vec(c1_ϕ[:,3]), c="r", marker="D")
PyPlot.scatter3D(vec(c2_ϕ[:,1]), vec(c2_ϕ[:,2]), vec(c2_ϕ[:,3]), c="m", marker="o")
PyPlot.xlim([-0.05,1.05])
PyPlot.ylim([-0.05,1.05])
PyPlot.zlim([-z_lim - 0.05, z_lim + 0.05])
PyPlot.title("Kernel Hilbert Space")
PyPlot.xlabel("Dimension 1") 
PyPlot.ylabel("Dimension 2")
PyPlot.zlabel("Dimension 3")

# Add Hyperplane
X = [ 0.0    0.0   ;  xy_mid xy_mid]
Y = [ xy_mid xy_mid;  0.0    0.0   ]
Z = [-z_lim  z_lim ; -z_lim  z_lim ]
PyPlot.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, color="b")

# Add Circular Intersection
v = linspace(0, 2π, 360)
X = xy_mid*(cos(v).+1)/2
Y = xy_mid .- X
Z = √2 * sqrt(X) .* sqrt(Y) .* sign(sin(v))
PyPlot.plot(X, Y, Z, color="k", ls="-")

# Kernel Geometry
U = generate_circles((0.0,0.0),ring[2],60,2)
W = vcat([ϕ(U[i,:]...) for i = 1:size(U,1)]...)
X, Y, Z = coordinate_matrices(W, 60, 2)
PyPlot.figure("KernelGeometry")
PyPlot.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, linewidth=0, 
                    cmap=ColorMap("coolwarm"))
PyPlot.xlim([-0.05,1.05])
PyPlot.ylim([-0.05,1.05])
PyPlot.zlim([-z_lim - 0.05, z_lim + 0.05])
PyPlot.title("Kernel Geometry")
PyPlot.xlabel("Dimension 1") 
PyPlot.ylabel("Dimension 2")
PyPlot.zlabel("Dimension 3")

# Add Circular Intersection
v = linspace(0, 2π, 360)
X = xy_mid*(cos(v).+1)/2
Y = xy_mid .- X
Z = √2 * sqrt(X) .* sqrt(Y) .* sign(sin(v))
PyPlot.plot(X, Y, Z, color="k", ls="-")

# Add Intersection to original space
PyPlot.figure("FeatureSpace")
PyPlot.plot(sqrt(xy_mid)*sin(v), sqrt(xy_mid)*cos(v), c="b", ls="-")


#===================================================================================================
  Kernel Trick Visualization 
===================================================================================================#

for (κ, title, n_points, n_contours, center, radius) in (
        (SquaredDistanceKernel(0.5), "Squared-Distance Kernel", 40, 15, (0.0, 0.0), 1.0),
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
    PyPlot.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=1.0, linewidth=0, 
                        cmap=ColorMap("coolwarm"))
    PyPlot.title(title)
    PyPlot.xlabel("Component 1") 
    PyPlot.ylabel("Component 2")
    PyPlot.zlabel("Component 3")
end
