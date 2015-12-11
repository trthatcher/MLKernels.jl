using MLKernels

GaussianKernel()

SquaredDistanceKernel()

ϕ = ExponentialClass(2.0, 1.0);
κ = SquaredDistanceKernel(1.0);
ψ = ϕ ∘ κ   # use \circ for '∘'

ismercer(κ)
isnegdef(κ)
ismercer(ψ)
isnegdef(ψ)

