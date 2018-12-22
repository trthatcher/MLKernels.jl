# Kernels

| Kernel | Mercer | Negative Definite | Stationary | Isotropic |
| --- | :-: | :-: | :-: | :-: |
| [Exponential Kernel](#Exponential-Kernel-1) | ✓ | | ✓ | ✓ |
| [Rational Quadratic Kernel](#Rational-Quadratic-Kernel-1) | ✓ | | ✓ | ✓ |
| [Exponentiated Kernel](#Exponentiated-Kernel-1) | ✓ | | | |

## Exponential Kernel

**Exponential Kernel**

The exponential kernel (see [`ExponentialKernel`](@ref)) is an isotropic Mercer kernel of
the form:

```math
\kappa(\mathbf{x},\mathbf{y})
= \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||\right)
\qquad \alpha > 0
```
where ``\alpha`` is a positive scaling parameter of the Euclidean distance. This kernel may
also be referred to as the Laplacian kernel (see [`LaplacianKernel`](@ref)).

**Squared-Exponential Kernel**

A similar form of the exponential kernel squares the Euclidean distance:

```math
\kappa(\mathbf{x},\mathbf{y})
= \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^2\right)
\qquad \alpha > 0
```
In this case, the kernel is often referred to as the squared exponential kernel (see
[`SquaredExponentialKernel`](@ref)) or the Gaussian kernel (see [`GaussianKernel`](@ref)).

**``\gamma``-Exponential Kernel**

Both the exponential and the squared exponential kernels are specific cases of the more
general ``\gamma``-exponential kernel:

```math
\kappa(\mathbf{x},\mathbf{y})
= \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^{2\gamma}\right)
\qquad \alpha > 0, \;\; 0 < \gamma \leq 1
```
where ``\gamma`` is an additional shape parameter of the Euclidean distance.

### Interface

```@docs
ExponentialKernel
SquaredExponentialKernel
GammaExponentialKernel
LaplacianKernel
GaussianKernel
RadialBasisKernel
```

## Rational-Quadratic Kernel

**Rational-Quadratic Kernel**

The rational-quadratic kernel (see [`RationalQuadraticKernel`](@ref)) is an isotropic
Mercer kernel given by the formula:

```math
\kappa(\mathbf{x},\mathbf{y})
= \left(1 +\alpha ||\mathbf{x} - \mathbf{y}||^{2}\right)^{-\beta}
\qquad \alpha > 0, \;\; \beta > 0
```
where ``\alpha`` is a positive scaling parameter and ``\beta`` is a shape parameter of the
Euclidean distance.

**``\gamma``-Rational-Quadratic Kernel**

The rational-quadratic kernel is a special case with ``\gamma = 1`` of the more general
``\gamma``-rational-quadratic kernel (see [`GammaRationalQuadraticKernel`](@ref)):

```math
\kappa(\mathbf{x},\mathbf{y})
= \left(1 +\alpha ||\mathbf{x} - \mathbf{y}||^{2\gamma}\right)^{-\beta}
\qquad \alpha > 0, \; \beta > 0, \; 0 < \gamma \leq 1
```
where ``\alpha`` is a positive scaling parameter, ``\beta`` is a positive shape parameter
and ``\gamma`` is a shape parameter of the Euclidean distance.

### Interface
```@docs
RationalQuadraticKernel
GammaRationalQuadraticKernel
```

## Exponentiated Kernel

The exponentiated kernel (see [`ExponentiatedKernel`](@ref)) is a Mercer kernel given by:

```math
\kappa(\mathbf{x},\mathbf{y}) = \exp\left(a \mathbf{x}^\intercal \mathbf{y} \right)
\qquad a > 0
```

where ``\alpha`` is a positive shape parameter.

### Interface
```@docs
ExponentiatedKernel
```

## Matern Kernel

The Matern kernel is a Mercer kernel given by:

```math
\kappa(\mathbf{x},\mathbf{y}) =
\frac{1}{2^{\nu-1}\Gamma(\nu)}
\left(\frac{\sqrt{2\nu}||\mathbf{x}-\mathbf{y}||}{\theta}\right)^{\nu}
K_{\nu}\left(\frac{\sqrt{2\nu}||\mathbf{x}-\mathbf{y}||}{\theta}\right)
```
where ``\nu`` and ``\rho`` are positive shape parameters.

### Interface
```@docs
MaternKernel
```

## Polynomial Kernel
The polynomial kernel is a Mercer kernel given by:

```math
\kappa(\mathbf{x},\mathbf{y}) =
(a \mathbf{x}^\intercal \mathbf{y} + c)^d
\qquad \alpha > 0, \; c \geq 0, \; d \in \mathbb{Z}_{+}
```
where ``a`` is a positive scale parameter, ``c`` is a non-negative shape parameter and ``d``
is a shape parameter that determines the degree of the resulting polynomial.

### Interface
```@docs
PolynomialKernel
```

## Periodic Kernel
The periodic kernel is given by:

```math
\kappa(\mathbf{x},\mathbf{y}) =
\exp\left(-\alpha \sum_{i=1}^n \sin(x_i - y_i)^2\right)
\qquad \alpha > 0
```
where ``a`` is a positive scale parameter.

### Interface
```@docs
PeriodicKernel
```

## Power Kernel
The power kernel is given by:

```math
\kappa(\mathbf{x},\mathbf{y}) =
\|\mathbf{x} - \mathbf{y} \|^{2\gamma}
\qquad \gamma \in (0,1]
```
where ``\gamma`` is a shape parameter of the Euclidean distance.

### Interface
```@docs
PowerKernel
```

## Log Kernel
The log kernel is a negative definite kernel given by:

```math
\kappa(\mathbf{x},\mathbf{y}) =
\log \left(1 + \alpha\|\mathbf{x} - \mathbf{y} \|^{2\gamma}\right)
\qquad \alpha > 0, \; \gamma \in (0,1]
```
where ``\alpha`` is a positive scaling parameter and ``\gamma`` is a shape parameter.

### Interface
```@docs
LogKernel
```

## Sigmoid Kernel
The Sigmoid Kernel is given by:

```math
\kappa(\mathbf{x},\mathbf{y}) =
\tanh(a \mathbf{x}^\intercal \mathbf{y} + c)
\qquad \alpha > 0, \; c \geq 0
```
The sigmoid kernel is a not a true kernel, although it has been used in application.

```@docs
SigmoidKernel
```