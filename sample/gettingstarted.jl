using MLKernels

# Constructing Standard Kernels

GaussianKernel()
SquaredDistanceKernel()

ϕ = ExponentialClass(2.0, 1.0);
κ = SquaredDistanceKernel(1.0);
ψ = ϕ ∘ κ   # use \circ for '∘'

ismercer(κ)
isnegdef(κ)
ismercer(ψ)
isnegdef(ψ)

w = round(rand(5),3);
ARD(κ, w)

# Kernel Functions

x = rand(3); y = rand(3);
kernel(ψ, x[1], y[1])
kernel(ψ, x, y)

X = rand(5,3);
kernelmatrix(ψ, X)
kernelmatrix(ψ, X', true)

Y = rand(4,3);
kernelmatrix(ψ, X, Y)
kernelmatrix(ψ, X', Y', true)

kernelmatrix(ψ, [X; Y])[1:5, 6:9]

# Kernel Operations

κ1 = GaussianKernel();

2k1 + 3

κ2 = PolynomialKernel();

κ1 + κ2

κ1 * κ2

κ3 = SineSquaredKernel();

κ3^0.5

κ4 = 2ScalarProductKernel() + 3;

κ4^3

exp(κ4)

tanh(κ4)
