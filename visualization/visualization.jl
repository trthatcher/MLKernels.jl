function generate_surface_rect!{T<:AbstractFloat}(
        lower::Tuple{T,T},
        upper::Tuple{T,T},
        dims::Tuple{Int64,Int64},
        X::Matrix{T} = Array(T, prod(dims), 2)
    )
    y_step = (upper[2] - lower[2])/(dims[1]-1)
    x_step = (upper[1] - lower[1])/(dims[2]-1)
    for j = 1:dims[2], i = 1:dims[1]
        idx = (j-1)*dims[1] + i
        X[idx, 1] = lower[1] + (j-1)*x_step
        X[idx, 2] = upper[2] - (i-1)*y_step
    end
    X
end

function kpca_project{T<:Base.LinAlg.BlasReal}(K::Matrix{T})
    Λ, V = LAPACK.syev!('V', 'U', centerkernelmatrix!(copy(K)))
    W = V[:,end:-1:end-2]
    μ = mean(K,2)
    Z = (K .- μ)* W
    Z = Z .- minimum(Z,1)  # Translate to (0,0,0)
    scale!(Z, 1 ./ vec(maximum(Z,1)))  # scale to fit in unit square
end

using MLKernels

n_x = 25
n_y = 25

using PyPlot
for (string, kern, lower, upper) in (("test2.svg", GaussianKernel(), (-3.0,-3.0), (3.0,3.0)),)
                                     #("test3.svg", PolynomialKernel(), (-2.0, -2.0), (2.0, 2.0)),
                                     #("test4.svg", RationalQuadraticKernel(), (-2.0, -2.0), (2.0, 2.0)))
    K = kernelmatrix(kern, generate_surface_rect!(lower, upper, (n_y, n_x)))
    W = kpca_project(K)
    X = reshape(W[:,1], (n_y, n_x))
    Y = reshape(W[:,2], (n_y, n_x))
    Z = reshape(W[:,3], (n_y, n_x))
    PyPlot.plot_wireframe(X, Y, Z)
end




#=
using Compose

X = generate_surface_rect!((0.0,0.0), (1.0,1.0), (n_y, n_x))
X_points = [(X[j*n_y + i,1], X[j*n_y + i,2]) for i = 1:n_y, j = 0:n_x-1]
lines = [[line(vec(X_points[:,i])) for i = 1:n_x];
         [line(vec(X_points[i,:])) for i = 1:n_y]]
draw(SVG("test1.svg", 7inch, 7inch), compose(context(), stroke("black"), lines...))

for (string, kernel, lower, upper) in (("test2.svg", GaussianKernel(), (-5.0,-5.0), (5.0,5.0)),
                                       ("test3.svg", PolynomialKernel(), (-2.0, -2.0), (2.0, 2.0)),
                                       ("test4.svg", RationalQuadraticKernel(), (-2.0, -2.0), (2.0, 2.0)))
    K = kernelmatrix(kernel, generate_surface_rect!(lower, upper, (n_y, n_x)))
    Z = kpca_project(K)
    Z_points = [(Z[j*n_y + i,1], Z[j*n_y + i,2]) for i = 1:n_y, j = 0:n_x-1]
    lines = [[line(vec(Z_points[:,i])) for i = 1:n_x];
             [line(vec(Z_points[i,:])) for i = 1:n_y]]
    draw(SVG(string, 7inch, 7inch), compose(context(), stroke("black"), lines...))
end

=#
#=
using MLKernels

K = kernelmatrix(ExponentiatedKernel(ScalarProductKernel()), X)

Λ, V = LAPACK.syev!('V', 'U', centerkernelmatrix!(copy(K)))

W = V[:,end:-1:end-1]

μ = mean(K,2)

Z = (K .- μ)* W

Z = Z .- minimum(Z,1)

scale!(Z, 1 ./ vec(maximum(Z,1)))


using Compose

#line(V[:,2][2:19])))

=#
