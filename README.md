# Machine Learning Kernels

[![Build Status](https://travis-ci.org/trthatcher/MLKernels.jl.svg?branch=master)](https://travis-ci.org/trthatcher/MLKernels.jl)
[![Coverage Status](https://coveralls.io/repos/trthatcher/MLKernels.jl/badge.svg)](https://coveralls.io/r/trthatcher/MLKernels.jl)

**MLKernels.jl** is a Julia package for kernel functions (or covariance functions in Gaussian 
processes) that are used in the kernel methods of machine learning. The goal is to provide a Julia
datatype for machine learning kernels and an efficient set of methods to calculate or approximate 
kernel matrices. The package has no dependencies beyond base Julia.

 - **Documentation:** http://mlkernels.readthedocs.org/

### Getting Started

Consistent with traditional literature on kernels, kernels come in two flavours:
 - **Mercer Kernels** (Continuous Positive Definite Kernels)
 - **Negative Definite Kernels**

Negative definite kernels are equivalent to conditionally positive definite kernels that are often found in Machine Learning literature. To convert a negative definite kernel to a conditionally positive definite kernel, simply multiply the result of the kernel function by -1.

Kernels are further broken into three main types:

 - **Base Kernels**: These are simple kernels that serve as building blocks for more complex kernels. They are easily extended.
 - **Composite Kernels**: These kernels are a scalar transformation of a Base Kernel. As a result, they are not standalone; they require a base kernel. Most kernels with widespread usage fall into this category.
 - **Combination Kernels**: These kernels are the result of addition or multiplication of Base or Composite kernels.

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
\**Not a true kernel*

Base Kernels are available as **Automatic Relevance Determination** (ARD) Kernels which act as a separate scaling constant for each element-wise operation on the inputs. For the dot product and the squared distance kernel, this corresponds to a linear scaling of each of the dimensions.

The `Kernel` data type is parametric - either `Float32`, `Float64` or `BigFloat` depending on the input arguments. The default is `Float64`. A kernel can be constructed using one of the many predefined kernels. Once a kernel has been constructed, it can be passed to the `kernel` function and used to compute kernel function of two vectors. For example, the simplest base kernel is the scalar (dot) product kernel:

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

One key property of kernels is whether or not they are Mercer kernels or negative definite kernels. The functions `ismercer` and `isnegdef` can be used to test whether or not a kernel is Mercer or negative definite:

```julia
julia> ismercer(SquaredDistanceKernel())
false

julia> isnegdef(SquaredDistanceKernel())
true

julia> ismercer(ScalarProductKernel())
true

julia> isnegdef(ScalarProductKernel())
false
```

Note that the definity of a base kernel does not imply anything about the definity of the composite kernel. For example, the Gaussian Kernel is Mercer but the underlying Squared Distance base kernel is negative definite:

```julia
julia> ismercer(GaussianKernel())
true

julia> isnegdef(GaussianKernel())
false
```
