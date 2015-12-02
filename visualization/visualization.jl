x_lower = -1.0
x_upper =  1.0
x_step  =  0.05

y_lower = -1.0
y_upper =  1.0
y_step  =  0.05

X_points = [(x, y) for y=y_upper:-y_step:y_lower, x = x_lower:x_step:x_upper]

n_y, n_x = size(X_points)

X = Float64[X_points[i][j] for i = 1:length(X_points), j = 1:2]

#E = vcat([vec([((x-1)*ny+y, x*ny+y) for y=1:ny, x=1:nx-1]);         # Left-to-Right Edges
#          vec([((x-1)*ny+y, (x-1)*ny+y+1) for y=1:ny-1, x=1:nx])])  # Top-to-Bottom Edges

#V = hcat(V...)'
#E = hcat(E...)'

#using Gadfly

using MLKernels

K = kernelmatrix(ExponentiatedKernel(ScalarProductKernel()), X)

Λ, V = LAPACK.syev!('V', 'U', centerkernelmatrix!(copy(K)))

W = V[:,end:-1:end-1]

μ = mean(K,2)

Z = (K .- μ)* W

Z = Z .- minimum(Z,1)

scale!(Z, 1 ./ vec(maximum(Z,1)))

Z_points = [(Z[j*n_y + i,1], Z[j*n_y + i,2]) for i = 1:n_y, j = 0:n_x-1]

using Compose

lines = [[line(vec(Z_points[:,i])) for i = 1:n_x];
         [line(vec(Z_points[i,:])) for i = 1:n_y]]

draw(SVG("Test.svg",7inch,7inch), compose(context(), stroke("black"), lines...))
#line(V[:,2][2:19])))


