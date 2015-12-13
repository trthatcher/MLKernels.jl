# Machine Learning Kernels

[![Documentation Status](https://readthedocs.org/projects/mlkernels/badge/?version=latest)](http://mlkernels.readthedocs.org/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/trthatcher/MLKernels.jl.svg?branch=master)](https://travis-ci.org/trthatcher/MLKernels.jl)
[![Coverage Status](https://coveralls.io/repos/trthatcher/MLKernels.jl/badge.svg)](https://coveralls.io/r/trthatcher/MLKernels.jl)

**MLKernels.jl** is a Julia package for Mercer kernel functions (or the 
covariance functions used in Gaussian processes) that are used in the kernel 
methods of machine learning. This package provides a flexible datatype for 
representing and constructing machine learning kernels as well as an efficient
set of methods to compute or approximate kernel matrices. The package has no 
dependencies beyond base Julia.

<p align="center"><img alt="Original Data" src="example/img/original.png"  /></p>

Through the use of kernel functions, kernel-based methods may operate in a high
(potentially infinite) dimensional implicit feature space without explicitly
mapping data from the original feature space to the new feature space. For
example, through the use of Kernel PCA, the non linearly separable data above
may be projected from the implicit kernel feature space onto a lower dimensional
space:

<p align="center"><img alt="Transformed Data" src="example/img/wireframe.png"  /></p>

This allows for the data to be linearly separated using a hyperplane:

<p align="center"><img alt="Separating Hyperplane" src="example/img/separatinghyperplane.png"  /></p>

The above plots were generated using PyPlot.jl and the code is available in
visualization.jl.

## Getting Started

### Constructing Kernels

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


#### Kernel Operations
