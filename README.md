# Machine Learning Kernels

[![Documentation Status](https://readthedocs.org/projects/mlkernels/badge/?version=latest)](http://mlkernels.readthedocs.org/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/trthatcher/MLKernels.jl.svg?branch=master)](https://travis-ci.org/trthatcher/MLKernels.jl)
[![Coverage Status](https://coveralls.io/repos/trthatcher/MLKernels.jl/badge.svg)](https://coveralls.io/r/trthatcher/MLKernels.jl)

#### Summary

**MLKernels.jl** is a Julia package for Mercer kernel functions (or the 
covariance functions used in Gaussian processes) that are used in the kernel 
methods of machine learning. This package provides a flexible datatype for 
representing and constructing machine learning kernels as well as an efficient
set of methods to compute or approximate kernel matrices. The package has no 
dependencies beyond base Julia.

#### Visualization

Through the use of kernel functions, kernel-based methods may operate in a high
(potentially infinite) dimensional implicit feature space without explicitly
mapping data from the original feature space to the new feature space. For 
example, through the use of Kernel PCA, non linearly separable data may be 
mapped to another space where it is linearly separable:

<p align="center"><img alt="Original Data" src="example/img/original.png"  /></p>
<p align="center"><img alt="Transformed Data" src="example/img/wireframe.png"  /></p>

This allows for the transformed data to be linearly separated using a 
hyperplane:

<p align="center"><img alt="Separating Hyperplane" src="example/img/separatinghyperplane.png"  /></p>

The above plots were generated using
[PyPlot.jl](https://github.com/stevengj/PyPlot.jl). The visualization code is
available in [visualization.jl](/example/visualization.jl).

## Getting Started ([example.jl](example/example.j))

#### Constructing Kernels

**MLKernels.jl** comes with a number of pre-defined kernel functions. For 
example, one of the most popular kernels is the Gaussian kernel (also known as 
the radial basis kernel or squared exponential covariance function). The code 
documentation may be used to learn more about the parametric forms of kernels 
using the `?` command and searching for the kernel name:

```julia
julia> using MLKernels

help?> GaussianKernel
search: GaussianKernel

  GaussianKernel(α) = exp(-α⋅‖x-y‖²)
```
The Gaussian Kernel has one scaling parameter `α`. We may instantiate the kernel
using:

```julia
julia> GaussianKernel(2.0)
KernelComposition{Float64}(ϕ=Exponential(α=2.0,γ=1.0),κ=SquaredDistance(t=1.0))
```

The `Kernel` data type is parametric - any subtype of `AbstractFloat`, though
only `Float32` and `Float64` are recommended. The default type is `Float64`.

The Gaussian kernel is actually a specific case of a more general class of
kernel. It is composition of scalar function and the squared (Euclidean) 
distance, `SquaredDistanceKernel`, which itself is a kernel. The scalar function
 referenced is referred to as the `ExponentialClass`, a subtype of the 
`CompositionClass` type. Composition classes may be composed with a kernel to 
yield a new kernel using the `∘` operator (shorthand for `KernelComposition`):

```julia
julia> ϕ = ExponentialClass(2.0, 1.0);

julia> κ = SquaredDistanceKernel(1.0);

julia> ψ = ϕ ∘ κ   # use \circ for '∘'
KernelComposition{Float64}(ϕ=Exponential(α=2.0,γ=1.0),κ=SquaredDistance(t=1.0))
```

**MLKernels.jl** implements only symmetric real-valued continuous kernel 
functions (a subset of the kernels studied in the literature). These kernels 
fall into two groups:
 - *Mercer Kernels* (positive definite kernels)
 - *Negative Definite Kernels*

A negative definite kernels is equivalent to the conditionally positive definite
kernels that are often discussed in machine learning literature. A conditionally
positive definite kernel is simply the negation of a negative definite kernel.

Returning to the example, the squared distance kernel is not a Mercer kernel 
although the resulting Gaussian kernel *is* Mercer. Kernels may be inspected 
using the `ismercer` and `isnegdef` functions:

```julia
julia> ismercer(κ)
false

julia> isnegdef(κ)
true

julia> ismercer(ψ)
true

julia> isnegdef(ψ)
false
```

`AdditiveKernel` types are available as **Automatic Relevance Determination** 
(ARD) Kernels. Weights may be applied to each element-wise function applied to
the input vectors. For the scalar product and squared distance kernel, this 
corresponds to a linear scaling of each of the dimensions.

```julia
julia> w = round(rand(5),3);

julia> ARD(κ, w)
ARD{Float64}(κ=SquaredDistance(t=1.0),w=[0.358,0.924,0.034,0.11,0.21])
```

#### Kernel Functions

The `kernel` function may be used to evaluate the kernel function for a pair of
scalar values or a pair of vectors for a given kernel:

```julia
julia> x = rand(3); y = rand(3);

julia> kernel(ψ, x[1], y[1])
0.9111186233655084

julia> kernel(ψ, x, y)
0.4149629934770782
```

Given a data matrix `X` (one observation/input per row), the kernel matrix
([Gramian matrix](https://en.wikipedia.org/wiki/Gramian_matrix) in the implicit 
feature space) for the set of vectors may be computed using the `kernelmatrix`
function:

```julia
julia> X = rand(5,3);

julia> kernelmatrix(ψ, X)
5x5 Array{Float64,2}:
 1.0       0.183858  0.511987  0.438051  0.265336
 0.183858  1.0       0.103226  0.345788  0.193466
 0.511987  0.103226  1.0       0.25511   0.613017
 0.438051  0.345788  0.25511   1.0       0.115458
 0.265336  0.193466  0.613017  0.115458  1.0     
```

where `kernelmatrix(ψ, X)[i,j]` is `kernel(ψ, vec(X[i,:]), vec(X[j,:]))`. If the
data matrix has one observation/input per column instead, an additional argument
`is_trans` may be supplied to instead operate on the columns:

```julia
julia> kernelmatrix(ψ, X', true)
5x5 Array{Float64,2}:
 1.0       0.183858  0.511987  0.438051  0.265336
 0.183858  1.0       0.103226  0.345788  0.193466
 0.511987  0.103226  1.0       0.25511   0.613017
 0.438051  0.345788  0.25511   1.0       0.115458
 0.265336  0.193466  0.613017  0.115458  1.0     
```

Similarly, given two data matrices, `X` and `Y`, the pairwise kernel matrix may
be computed using:

```julia
julia> kernelmatrix(ψ, X, Y)
5x4 Array{Float64,2}:
 0.805077  0.602521  0.166728  0.503091
 0.110309  0.121443  0.487499  0.711981
 0.692844  0.987933  0.24032   0.404972
 0.505462  0.30802   0.114626  0.641405
 0.248455  0.580033  0.678817  0.423386

julia> kernelmatrix(ψ, X', Y', true)
5x4 Array{Float64,2}:
 0.805077  0.602521  0.166728  0.503091
 0.110309  0.121443  0.487499  0.711981
 0.692844  0.987933  0.24032   0.404972
 0.505462  0.30802   0.114626  0.641405
 0.248455  0.580033  0.678817  0.423386
```

This is equivalent to the upper right corner of the kernel matrix of `[X;Y]`:

```julia
julia> kernelmatrix(ψ, [X; Y])[1:5, 6:9]
5x4 Array{Float64,2}:
 0.805077  0.602521  0.166728  0.503091
 0.110309  0.121443  0.487499  0.711981
 0.692844  0.987933  0.24032   0.404972
 0.505462  0.30802   0.114626  0.641405
 0.248455  0.580033  0.678817  0.423386
```

#### Kernel Operations

All kernels may be translated and scaled by positive real numbers. Scaling or
translating a `Kernel` object will create a `KernelAffinity` object:

```julia
julia> κ1 = GaussianKernel();

julia> 2κ1 + 3
Affine{Float64}(a=2.0,c=3.0,κ=KernelComposition{Float64}(ϕ=Exponential(α=1.0,γ=1.0),κ=SquaredDistance(t=1.0)))
```

*Mercer* and *negative definite kernels* may be added together to construct a
`KernelSum` object that may be used like any other kernel:

```julia
julia> κ2 = PolynomialKernel();

julia> κ1 + κ2
KernelSum{Float64}(KernelComposition(ϕ=Exponential(α=1.0,γ=1.0),κ=SquaredDistance(t=1.0)), KernelComposition(ϕ=Polynomial(a=1.0,c=1.0,d=3),κ=ScalarProduct()))
```

*Mercer kernels* may also be multiplied together to create a `KernelProduct`
object:

```julia
julia> κ1 * κ2
KernelProduct{Float64}(KernelComposition(ϕ=Exponential(α=1.0,γ=1.0),κ=SquaredDistance(t=1.0)), KernelComposition(ϕ=Polynomial(a=1.0,c=1.0,d=3),κ=ScalarProduct()))
```

Several other operations are defined that act as shortcuts for pre-defined
kernels:

```julia
julia> κ3 = SineSquaredKernel();

julia> κ3^0.5
KernelComposition{Float64}(ϕ=Power(a=1.0,c=0.0,γ=0.5),κ=SineSquared(p=3.141592653589793,t=1.0))

julia> κ4 = 2ScalarProductKernel() + 3;

julia> κ4^3
KernelComposition{Float64}(ϕ=Polynomial(a=2.0,c=3.0,d=3),κ=ScalarProduct())

julia> exp(κ4)
KernelComposition{Float64}(ϕ=Exponentiated(a=2.0,c=3.0),κ=ScalarProduct())

julia> tanh(κ4)
KernelComposition{Float64}(ϕ=Sigmoid(a=2.0,c=3.0),κ=ScalarProduct())
```

#### Documentation

Further [**Documentation**](http://http://mlkernels.readthedocs.org/en/latest/)
is available on Read the Docs. 
