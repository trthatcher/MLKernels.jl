# Machine Learning Kernels

[![Build Status](https://travis-ci.org/trthatcher/MLKernels.jl.svg?branch=master)](https://travis-ci.org/trthatcher/MLKernels.jl)
[![Coverage Status](https://coveralls.io/repos/trthatcher/MLKernels.jl/badge.svg)](https://coveralls.io/r/trthatcher/MLKernels.jl)

MLKernels.jl is a Julia package for Mercer and non-Mercer kernels that are used in the kernel methods of machine learning. The goal is to provide a Julia datatype for machine learning kernels and an efficient set of methods to calculate or approximate kernel matrices. The package has no dependencies beyond base Julia.

Consistent with traditional literature on kernels, kernels come in two flavours:
 - Positive Definite Kernels (Mercer Kernels)
 - Negative Definite Kernels

Negative definite kernels are equivalent to conditionally positive definite kernels that are often found in Machine Learning literature. To convert a negative definite kernel to a conditionally positive definite kernel, simply multiply the result of the kernel function by -1.

Kernels are further broken into three categories:
 - Base Kernels: These are simple kernels that serve as building blocks for more complex kernels. They are easily extended.
 - Composite Kernels: These kernels are a scalar transformation of the result of a Base Kernel. As a result, they are not standalone; they require a base kernel. Most kernels experiencing widespread usage fall into this category.
 - Combination Kernels: These kernels are the result of addition or multiplication of Base or Composite kernels.

The following table outlines all documented kernel combinations available:

<table>
  <tr>
    <th colspan="2" rowspan="3">Composite Kernels</th>
    <th colspan="5">Base Kernels</th>
  </tr>
  <tr>
    <td colspan="3">Negative Definite Kernels</td>
    <td colspan="2">Mercer Kernels</td>
  </tr>
  <tr>
    <td>Squared Distance</td>
    <td>Chi Squared</td>
    <td>Sine Squared</td>
    <td>Scalar Product</td>
    <td>Mercer Sigmoid</td>
  </tr>
  <tr>
    <td rowspan="2">Negative Definite Kernels</td>
    <td>Power Kernel</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Log Kernel</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="5">Mercer Kernels</td>
    <td>Exponential Kernel</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Rational Quadratic Kernel</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Matern Kernel</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Polynomial Kernel</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>ExponentiatedKernel</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Non-Kernels</td>
    <td>Sigmoid Kernel</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

All kernels are available as Automatic Relevance Determination (ARD) Kernels which act as a separate scaling constant for each dimension of input.

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

## References

[Marc G. Genton. 2002. Classes of kernels for machine learning: a statistics perspective. J. Mach. Learn. Res. 2 (March 2002), 299-312.](http://dl.acm.org/citation.cfm?id=944815)

[Petros Drineas and Michael W. Mahoney. 2005. On the Nyström Method for Approximating a Gram Matrix for Improved Kernel-Based Learning. J. Mach. Learn. Res. 6 (December 2005), 2153-2175.](http://dl.acm.org/citation.cfm?id=1194916)

C. K. I. Williams and M. Seeger. Using the Nyström method to speed up kernel machines. In Annual Advances in Neural Information Processing Systems 13: Proceedings of the 2000 Conference, pages 682-688, 2001.
