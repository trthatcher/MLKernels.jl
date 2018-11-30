# Kernels

| Kernel | Mercer | Negative Definite | Stationary | Isotropic |
| --- | :-: | :-: | :-: | :-: |
| [Exponential Kernel](#Exponential-Kernel-1) | ✓ | | ✓ | ✓ |
| [Rational Quadratic Kernel](#Rational-Quadratic-Kernel-1) | ✓ | | ✓ | ✓ |
| [Exponentiated Kernel](#Exponentiated-Kernel-1) | ✓ | | | |

## Exponential Kernel

The exponential kernel is an isotropic Mercer kernel of the form:

```math
\kappa(\mathbf{x},\mathbf{y})
= \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^{\gamma}\right)
\qquad \alpha > 0, \;\; 0 < \gamma \leq 1
```
where ``\alpha`` is a positive scaling parameter and ``\gamma`` is a positive shape
parameter of the Euclidean distance with a maximum of 1. When ``\gamma`` is not a fixed
parameter, the kernel is referred to as the ``\gamma``-exponential kernel in this package.

It is common to use ``\gamma = 1``. In this case, the kernel is typically referred to as
the squared exponential covariance function in the context of Gaussian Processes or the
Gaussian kernel (see [`GaussianKernel`](@ref)) in other applications. In machine learning
circles, it may also be known as the radial basis kernel (see [`RadialBasisKernel`](@ref)).

When ``\gamma = 1``, this kernel may also be referred to as the exponential covariance
function or the Laplacian kernel (see [`LaplacianKernel`](@ref)).

```@docs
ExponentialKernel
LaplacianKernel
SquaredExponentialKernel
GaussianKernel
RadialBasisKernel
GammaExponentialKernel
```

## Rational Quadratic Kernel
```@docs
RationalQuadraticKernel
GammaRationalQuadraticKernel
```

## Exponentiated Kernel
```@docs
ExponentiatedKernel
```

MaternKernel
LinearKernel
PolynomialKernel
PeriodicKernel

## Negative Definite Kernels
```@docs
```
PowerKernel
LogKernel

## Other Kernels

```@docs
```
SigmoidKernel