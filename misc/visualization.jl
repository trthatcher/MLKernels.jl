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
    Λ, V = LAPACK.syev!('V', 'U', copy(K))
    W = V[:,end:-1:end-2] * 10  # eigenvalues are in ascending order
end

using MLKernels

κ = GaussianKernel()
n_points = 40
n_contours = 15
U = generate_circles((0.0, 0.0), 1.0, n_points, n_contours)
W = kpca_3d(kernelmatrix(κ, U))
X, Y, Z = coordinate_matrices(W, n_points, n_contours)

