# Machine Learning Kernels

[![Build Status](https://travis-ci.org/trthatcher/MLKernels.jl.svg?branch=master)](https://travis-ci.org/trthatcher/MLKernels.jl)
[![Coverage Status](https://coveralls.io/repos/trthatcher/MLKernels.jl/badge.svg)](https://coveralls.io/r/trthatcher/MLKernels.jl)

MLKernels.jl is a Julia package for Mercer and negative definite kernels that are used in the kernel methods of machine learning. The goal is to provide a Julia datatype for machine learning kernels and an efficient set of methods to calculate or approximate kernel matrices. The package has no dependencies beyond base Julia.

Consistent with traditional literature on kernels, kernels come in two flavours:
 - Positive Definite Kernels (Mercer Kernels)
 - Negative Definite Kernels

Negative definite kernels are equivalent to conditionally positive definite kernels that are often found in Machine Learning literature. To convert a negative definite kernel to a conditionally positive definite kernel, simply multiply the result of the kernel function by -1.

Kernels are further broken into three categories:

 - Base Kernels: These are simple kernels that serve as building blocks for more complex kernels. They are easily extended.
 - Composite Kernels: These kernels are a scalar transformation of the result of a Base Kernel. As a result, they are not standalone; they require a base kernel. Most kernels experiencing widespread usage fall into this category.
 - Combination Kernels: These kernels are the result of addition or multiplication of Base or Composite kernels.

Base kernels can be instantiated on their own. The following chart illustrates the possible base kernel and composite kernel combinations:

<table>
  <tr>
    <th rowspan="2">Composite Kernels</th>
    <th align="center" colspan="5">Base Kernels</th>
  </tr>
  <tr>
    <td align="center">Squared Distance</td>
    <td align="center">Chi Squared</td>
    <td align="center">Sine Squared</td>
    <td align="center">Scalar Product</td>
    <td align="center">Mercer Sigmoid</td>
  </tr>
  <tr>
    <td align="center">Power Kernel</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">Log Kernel</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">Exponential Kernel</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">Rational Quadratic Kernel</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">Matern Kernel</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">Polynomial Kernel</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
  </tr>
  <tr>
    <td align="center">ExponentiatedKernel</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
  </tr>
  <tr>
    <td align="center">Sigmoid Kernel*</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
  </tr>
</table>
*Not a true kernel

Base Kernels are available as Automatic Relevance Determination (ARD) Kernels which act as a separate scaling constant for each element-wise operation on the inputs. For the dot product and the squared distance kernel, this corresponds to a linear scaling of each of the dimensions.

### Getting Started

A kernel can be constructed using one of the many predefined kernels. Once a kernel has been constructed, it can be passed to the `kernel` function and used to compute kernel function of two vectors. For example, the simplest base kernel is the scalar (dot) product kernel:

```julia
julia> κ = ScalarProductKernel()
ScalarProductKernel{Float64}()

julia> x,y = (10rand(3),10rand(3))
([4.5167,7.60119,0.692922],[7.46812,0.605204,9.39787])

julia> kernel(κ,x,y)
44.843518824558856

julia> dot(x,y)
44.843518824558856
```

Several other base kernels have been defined. For example, the squared distance kernel is the squared l2 norm:

```julia
julia> κ = SquaredDistanceKernel()
SquaredDistanceKernel{Float64}(t=1.0)

julia> kernel(κ,x,y)
133.43099175980925

julia> dot(x.-y,x.-y)
133.43099175980925
```
A subset of the base kernels, known as Additive Kernels, are also available as Automatic Relevance Determination kernels. If the kernel function consists of a sum of elementwise functions applied to each dimension, then it is amenable to automatic relevance determination. Continuing on with the squared distance kernel:

```julia
julia> SquaredDistanceKernel <: AdditiveKernel
true

julia> w = 10rand(3)
3-element Array{Float64,1}:
 9.29903
 1.24022
 3.21233

julia> ψ = ARD(κ,w)
ARD{Float64}(κ=SquaredDistanceKernel(t=1.0),w=[9.29903,1.24022,3.21233])

julia> kernel(ψ,x,y)
1610.469072503976

julia> dot((x.-y).*w,(x.-y).*w)
1610.4690725039757
```

Base kernels can be extended using composite kernels. These kernels are a function of a base kernel. For example, the Gaussian Kernel (Radial Basis Kernel) may be constructed in the following way:

```julia
julia> ϕ = ExponentialKernel(κ)
ExponentialKernel{Float64}(κ=SquaredDistanceKernel(t=1.0),α=1.0,γ=1.0)

julia> GaussianKernel()
ExponentialKernel{Float64}(κ=SquaredDistanceKernel(t=1.0),α=1.0,γ=1.0)
```

To compute a kernel matrix:

```
julia> X = rand(5,3);

julia> kernelmatrix(ϕ, X)
5x5 Array{Float64,2}:
 1.0       0.710224  0.353483  0.858427  0.704625
 0.710224  1.0       0.743461  0.799072  0.713864
 0.353483  0.743461  1.0       0.584813  0.284877
 0.858427  0.799072  0.584813  1.0       0.526931
 0.704625  0.713864  0.284877  0.526931  1.0     
```

This assumes that each row of X is an observation. If observations are stored as columns, use `'T'` (default is `'N'`) for the first argument:

```julia
julia> kernelmatrix(ϕ, X, 'T')
3x3 Array{Float64,2}:
 1.0       0.617104  0.245039
 0.617104  1.0       0.408998
 0.245039  0.408998  1.0   
```

### Reference

#### Base Kernels

<table>
  <tr>
    <th>Definity</th>
    <th>Type</th>
    <th>k(x,y)</th>
    <th>Restrictions</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td rowspan=3>Negative Definite Kernels</td>
    <td>SquaredDistanceKernel</td>
    <td>sum((x-y).^2t)</td>
    <td>t &gt; 0</td>
    <td>Uses BLAS for t = 1</td>
  </tr>
  <tr>
    <td>SineSquaredKernel</td>
    <td>sum(sin(x-y).^2t)</td>
    <td>t &gt; 0</td>
    <td></td>
  </tr>
  <tr>
    <td>ChiSquaredKernel</td>
    <td>sum(((x-y).^2 ./ (x+y)).^t)</td>
    <td>t &gt; 0</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan=2>Mercer Kernels</td>
    <td>ScalarProductKernel</td>
    <td>dot(x,y)</td>
    <td></td>
    <td>Uses BLAS</td>
  </tr>
  <tr>
    <td>MercerSigmoidKernel</td>
    <td>dot(tanh((x.-d)/b),tanh((y.-d)/b))</td>
    <td>t &gt; 0</td>
    <td>Uses BLAS</td>
  </tr>
</table>

#### Composite Kernels

The following table outlines all documented kernel combinations available:

<table>
  <tr>
    <th colspan="2" rowspan="2">Composite Kernels</th>
    <th align="center" colspan="5">Base Kernels</th>
  </tr>
  <tr>
    <td align="center">Squared Distance</td>
    <td align="center">Chi Squared</td>
    <td align="center">Sine Squared</td>
    <td align="center">Scalar Product</td>
    <td align="center">Mercer Sigmoid</td>
  </tr>
  <tr>
    <td align="center"rowspan="2">Negative Definite Kernels</td>
    <td align="center">Power Kernel</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">Log Kernel</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td rowspan="5">Mercer Kernels</td>
    <td align="center">Exponential Kernel</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">Rational Quadratic Kernel</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">Matern Kernel</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">Polynomial Kernel</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
  </tr>
  <tr>
    <td align="center">ExponentiatedKernel</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
  </tr>
  <tr>
    <td align="center">Non-Kernels</td>
    <td align="center">Sigmoid Kernel</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center">&#10004;</td>
    <td align="center">&#10004;</td>
  </tr>
</table>



New Kernels may be constructed by scaling and translating existing kernels by positive real numbers. Further, kernels may be arbitrarily added and multiplied together to create composite kernels.


The `Kernel` data type is parametric - either `Float32` or `Float64` depending on the input arguments. 

Not all of the provided kernels are Mercer kernels (a kernel is Mercer if its kernel matrices are positive semi-definite). The function `ismercer` will return `true` if the kernel is a Mercer kernel, and `false` otherwise.

Given a data matrix with one observation per row, the kernel matrix for kernel `κ` can be calculated using the `kernelmatrix` method:

## Convex Cone of Kernels

According to the properties of Mercer kernels:

- If κ is a kernel and a > 0, then aκ is also a kernel
- If κ₁ is a kernel and κ₂ is a kernel, then κ₁ + κ₂ is a kernel

In other words, Mercer Kernels form a convex cone. This package supports addition and multiplication of Kernel objects:

## Kernel Product

Mercer Kernels have the additional property:

- If κ₁ is a kernel and κ₂ is a kernel, then κ₁κ₂ is a kernel

This package allows for a scaled point-wise product of kernels to be defined using the `KernelProduct` type:

## Approximating Kernel Matrices

The Nystrom Method of approximating kernel matrices has been implemented. It requires an additional array of integers that specify the sampled columns. It should be noted that the Nystrom method is intended for large matrices:

#### Citations

[Marc G. Genton. 2002. Classes of kernels for machine learning: a statistics perspective. J. Mach. Learn. Res. 2 (March 2002), 299-312.](http://dl.acm.org/citation.cfm?id=944815)

[Petros Drineas and Michael W. Mahoney. 2005. On the Nyström Method for Approximating a Gram Matrix for Improved Kernel-Based Learning. J. Mach. Learn. Res. 6 (December 2005), 2153-2175.](http://dl.acm.org/citation.cfm?id=1194916)

C. K. I. Williams and M. Seeger. Using the Nyström method to speed up kernel machines. In Annual Advances in Neural Information Processing Systems 13: Proceedings of the 2000 Conference, pages 682-688, 2001.
