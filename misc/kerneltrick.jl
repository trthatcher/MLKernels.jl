#=
using Gadfly, Colors

n = 60

x = [round(x,1) for x in vcat(2*(randn(n)-7), 2*(randn(n)+7), 2*randn(n))]
y = zeros(n*3)
c = vcat(Int64[1 for i = 1:2n], Int64[2 for i = 1:n])

P = plot(
        x=x,
        colour= map(class -> "Class $class", c),
        Geom.histogram(bincount=n),
        Scale.color_discrete_manual(colorant"red",colorant"blue"),
        Guide.XLabel("Feature"),
        Guide.YLabel(""),
        Guide.title("Non-Linearly Separable Data"),
        Guide.colorkey("")
    )

draw(PNG("Feature.png", 6inch, 3inch), P)

P = plot(
        x=x,
        y=x.^2,
        colour= map(class -> "Class $class", c),
        Geom.point(),
        Scale.color_discrete_manual(colorant"red",colorant"blue"),
        Guide.XLabel("Component 1"),
        Guide.YLabel("Component 2"),
        Guide.title("Linearly Separable Data"),
        Guide.colorkey(""),
        yintercept=[7^2], 
        Geom.hline(color=colorant"black")
    )

draw(PNG("FeatureMap.png", 6inch, 4inch), P)

=#

function generate_grid{T<:AbstractFloat}(x_range::FloatRange{T}, y_range::FloatRange{T})
    X = Array(T, length(y_range), length(x_range))
    Y = Array(T, length(y_range), length(x_range))
    for j in eachindex(x_range), i in eachindex(y_range)
        X[i,j] = x_range[j]
        Y[i,j] = y_range[i]
    end
    (X,Y)
end

function kpca_2d{T<:Base.LinAlg.BlasReal}(K::Matrix{T})
    ΛV = eigfact(Symmetric(K))
    W = ΛV.vectors[:,end:-1:end-1]
end

using MLKernels

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

x1_range = -2.0:0.05:2.0
x2_range = x1_range
n_x1   = length(x1_range)
n_x2   = length(x2_range)
x1, x2 = generate_grid(x1_range, x2_range)

Z = generate_point_cluster(100, (0.0,0.0), 2.0)
X = hcat(vec(x1), vec(x2))

Kzz = kernelmatrix(GaussianKernel(), Z)
Kxz = kernelmatrix(GaussianKernel(), X, Z)

W = kpca_2d(Kzz)

Z_phi = Kzz * W
X_phi = Kxz * W

x1_phi   = reshape(X_phi[:,1], n_x1, n_x2)
x2_phi   = reshape(X_phi[:,2], n_x1, n_x2)
X_points = [(x1_phi[j], x2_phi[i]) for j = 1:n_x1, i = 1:n_x2]

using Compose

using Gadfly

lines = [[line(vec(X_points[:,i])) for i = 1:n_x1];
         [line(vec(X_points[i,:])) for i = 1:n_x2]]

#draw(SVG("Test.svg",7inch,7inch), compose(context(), stroke("black"), lines...))

plot(x = vec(Z_phi[:,1]), y = vec(Z_phi[:,2]), Geom.point,
     Guide.annotation(compose(context(), stroke("black"), lines...)))

#=
x_lower = -1.0
x_upper =  1.0
x_step  =  0.05

y_lower = -1.0
y_upper =  1.0
y_step  =  0.05

x_range = -1.0 : 0.05 : 1.0
y_range = -1.0 : 0.05 : 1.0

X_points = [(x, y) for y=y_upper:-y_step:y_lower, x = x_lower:x_step:x_upper]

n_y, n_x = size(X_points)

X = Float64[X_points[i][j] for i = 1:length(X_points), j = 1:2]

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
=#
