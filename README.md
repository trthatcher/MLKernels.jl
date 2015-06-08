# Machine Learning Kernels

[![Build Status](https://travis-ci.org/trthatcher/MLKernels.jl.svg?branch=master)](https://travis-ci.org/trthatcher/MLKernels.jl)
[![Coverage Status](https://coveralls.io/repos/trthatcher/MLKernels.jl/badge.svg)](https://coveralls.io/r/trthatcher/MLKernels.jl)

MLKernels.jl is a Julia package for Mercer and non-Mercer kernels that are used in the kernel methods of machine learning. The goal is to provide a Julia datatype for machine learning kernels and an efficient set of methods to calculate or approximate kernel matrices. The package has no dependencies beyond base Julia.

The package currently supports six pre-defined Mercer (positive definite) kernels:

- Exponential Class Kernels (includes the Gaussian/Radial Basis Kernel)
- Rational Quadratic Class Kernels
- Matern Kernel
- Polynomial Kernel
- Periodic Kernel

A "non-kernel" kernel:
- Sigmoid Kernel

Finally, two conditionally positive definite kernels are also included:
- Power Kernel
- Log Kernel

All kernels available as Automatic Relevance Determination (ARD) Kernels which act as a scaling constant for each dimension of input.

New Kernels may be constructed by scaling and translating existing kernels by positive real numbers. Further, kernels may be arbitrarily added and multiplied together to create composite kernels.

## Creating Basic Kernels

A number of standard kernels have been pre-defined. For example, to create a Polynomial Kernel object:

```julia
julia> κ = PolynomialKernel()
PolynomialKernel{Float64}(α=1.0,c=1.0,d=2)

julia> κ = PolynomialKernel(1.0f0)
PolynomialKernel{Float32}(α=1.0,c=1.0,d=2)
```

The `Kernel` data type is parametric - either `Float32` or `Float64` depending on the input arguments. 

If one wishes to see more information on the kernel, there is the function `description` which will print out a description of the kernel if it exists:

```julia
julia> description(κ)
 Polynomial Kernel:
 
 The polynomial kernel is a non-stationary kernel which represents
 the original features as in a feature space over polynomials up to 
 degree d of the original variables:

     k(x,y) = (αxᵗy + c)ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0, d > 0

 This kernel is sensitive to numerical instability in the case that
 d is increasingly large and αxᵗy + c approaches zero.
```

To evaluate a kernel for two values, one simply uses the `kernel` method with the first argument being the kernel of choice:

```julia
julia> x, y = [1.0, 2.0], [1.0, 1.0]
([1.0,2.0],[1.0,1.0])

julia> kernel(κ, x, y)
16.0
```

Note that the kernels are callable objects ("functors"), so the same as above may be accomplished through:

```julia
julia> PolynomialKernel()(x, y)
16.0

julia> κ(x, y)
16.0
```

Not all of the provided kernels are Mercer kernels (a kernel is Mercer if its kernel matrices are positive semi-definite). The function `ismercer` will return `true` if the kernel is a Mercer kernel, and `false` otherwise.

```julia
julia> ismercer(κ)
true
```

## Calculating Kernel Matrices

Given a data matrix with one observation per row, the kernel matrix for kernel `κ` can be calculated using the `kernelmatrix` method:

```julia
julia> X = rand(10, 5)
10x5 Array{Float64,2}:
 0.504367  0.00359014  0.33527   0.525783   0.309577 
 0.743592  0.00457371  0.594875  0.336426   0.102312 
 0.713523  0.84033     0.109831  0.0415291  0.830992 
 0.311136  0.312127    0.441631  0.484347   0.600404 
 0.756499  0.307774    0.473343  0.66326    0.63204  
 0.154361  0.174706    0.467842  0.504841   0.0189265
 0.72457   0.683054    0.675841  0.79224    0.252396 
 0.148947  0.704279    0.472177  0.263061   0.016687 
 0.953466  0.286904    0.386647  0.150966   0.863391 
 0.736272  0.497723    0.910921  0.58754    0.666643 

julia> kernelmatrix(κ, X)
10x10 Array{Float64,2}:
 3.02444  3.17932  2.8184   3.05077  …  4.36464  1.90285  3.83462   4.81293
 3.17932  4.12283  2.88572  2.95797     5.00105  2.20544  4.32404   5.55788
 2.8184   2.88572  8.52409  4.21006     5.79777  3.14976  7.2233    6.875  
 3.05077  2.95797  4.21006  3.93762     5.16367  2.59895  4.61591   6.10857
 4.35036  4.55104  5.7752   5.02302     7.63557  3.02052  6.96137   8.71661
 2.26994  2.45131  1.8091   2.44939  …  3.81068  2.25017  2.16309   3.74812
 4.36464  5.00105  5.79777  5.16367     9.85755  4.49745  6.17847   9.75201
 1.90285  2.20544  3.14976  2.59895     4.49745  3.27836  2.49881   4.22714
 3.83462  4.32404  7.2233   4.61591     6.17847  2.49881  8.46311   8.18696
 4.81293  5.55788  6.875    6.10857     9.75201  4.22714  8.18696  11.6228 

julia> kernel(κ, vec(X[1,:]), vec(X[1,:]))
3.02443612827351
```

Optimised `kernelmatrix` methods have been defined for the kernels listed earlier, in addition to a generic fall-back method. Therefore, it is preferable to use the `kernelmatrix` method where possible rather than explicitly calculating each entry in the kernel matrix.

## Convex Cone of Kernels

According to the properties of Mercer kernels:

- If κ is a kernel and a > 0, then aκ is also a kernel
- If κ₁ is a kernel and κ₂ is a kernel, then κ₁ + κ₂ is a kernel

In other words, Mercer Kernels form a convex cone. This package supports addition and multiplication of Kernel objects:

```julia
julia> 5 * ExponentialKernel()
KernelProduct{Float64}(5.0, ExponentialKernel(α=1.0,γ=1.0))

julia> PolynomialKernel() + SigmoidKernel()
KernelSum{Float64}(PolynomialKernel(α=1.0,c=1.0,d=2), SigmoidKernel(α=1.0,c=1.0))
```

Optimised methods for `kernelmatrix` have also been defined for `KernelProduct` and `KernelSum`.


## Kernel Product

Mercer Kernels have the additional property:

- If κ₁ is a kernel and κ₂ is a kernel, then κ₁κ₂ is a kernel

This package allows for a scaled point-wise product of kernels to be defined using the `KernelProduct` type:

```julia
julia> 3*PolynomialKernel()*SigmoidKernel()
KernelProduct{Float64}(3.0, PolynomialKernel(α=1.0,c=1.0,d=2), SigmoidKernel(α=1.0,c=1.0))
```

## Approximating Kernel Matrices

The Nystrom Method of approximating kernel matrices has been implemented. It requires an additional array of integers that specify the sampled columns. It should be noted that the Nystrom method is intended for large matrices:

```julia
julia> X = rand(5,3)
5x3 Array{Float64,2}:
 0.812068  0.644385  0.933447 
 0.243326  0.679563  0.0939346
 0.973343  0.864038  0.757773 
 0.215628  0.167303  0.256735 
 0.932313  0.592807  0.782866 

julia> kernelmatrix(ExponentialKernel(), X)
5x5 Array{Float64,2}:
 1.0       0.357191  0.900218  0.353     0.960988
 0.357191  1.0       0.365081  0.748502  0.384099
 0.900218  0.365081  1.0       0.269655  0.926928
 0.353     0.748502  0.269655  1.0       0.378513
 0.960988  0.384099  0.926928  0.378513  1.0     

julia> nystrom(ExponentialKernel(), X, [1, 3, 5])
5x5 Array{Float64,2}:
 1.0       0.357191  0.900218  0.353     0.960988
 0.357191  0.150183  0.365081  0.141043  0.384099
 0.900218  0.365081  1.0       0.269655  0.926928
 0.353     0.141043  0.269655  0.190468  0.378513
 0.960988  0.384099  0.926928  0.378513  1.0     
```

## References

[Marc G. Genton. 2002. Classes of kernels for machine learning: a statistics perspective. J. Mach. Learn. Res. 2 (March 2002), 299-312.](http://dl.acm.org/citation.cfm?id=944815)

[Petros Drineas and Michael W. Mahoney. 2005. On the Nyström Method for Approximating a Gram Matrix for Improved Kernel-Based Learning. J. Mach. Learn. Res. 6 (December 2005), 2153-2175.](http://dl.acm.org/citation.cfm?id=1194916)

C. K. I. Williams and M. Seeger. Using the Nyström method to speed up kernel machines. In Annual Advances in Neural Information Processing Systems 13: Proceedings of the 2000 Conference, pages 682-688, 2001.
