# Kernels implemented in MLKernels.jl

## Mercer kernels

### Exponential Class Kernels

The exponential kernel is a positive definite kernel defined as:

    k(x,y) = exp(-α‖x-y‖²ᵞ)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, γ ∈ (0,1]

Since the value of the function decreases as x and y differ, it can
be interpreted as a similarity measure. It is derived by exponentiating
the conditionally positive-definite power kernel.

When γ = 0.5, it is known as the Laplacian or exponential kernel. When
γ = 1, it is known as the Gaussian or squared exponential kernel.

---
Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian 
Processes for Machine Learning (Adaptive Computation and Machine 
Learning). The MIT Press.


### Rational Quadratic Class Kernels

The rational kernel is a stationary kernel that is similar in shape
to the Gaussian kernel:

    k(x,y) = (1 + α‖x-y‖²ᵞ)⁻ᵝ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, β > 0, γ ∈ (0,1]

It is derived by exponentiating the conditionally positive-definite log
kernel. Setting α = α'/β, it can be seen that the rational kernel 
converges to the gamma exponential kernel as β → +∞.

When γ = 1, the kernel is referred to as the rational quadratic kernel.

---
Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian 
Processes for Machine Learning (Adaptive Computation and Machine 
Learning). The MIT Press.


### Matern Kernel

    k(x,y) = ...    x ∈ ℝⁿ, y ∈ ℝⁿ, ν > 0, θ > 0

on , vol., no., pp.113,116, 6-6 July 2005


### Polynomial Kernel
 
The polynomial kernel is a non-stationary kernel which represents
the original features as in a feature space over polynomials up to 
degree d of the original variables:

    k(x,y) = (αxᵀy + c)ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0, d > 0

This kernel is sensitive to numerical instability in the case that
d is increasingly large and αxᵀy + c approaches zero.


## Conditionally Positive Definite Kernels

### Power Kernel

The power kernel (also known as the unrectified triangular kernel) is
a conditionally positive definite kernel. An important feature of the
power kernel is that it is scale invariant. The function is given by:

    k(x,y) = -‖x-y‖²ᵞ   x ∈ ℝⁿ, y ∈ ℝⁿ, γ ∈ (0,1]

---
Boughorbel, S.; Tarel, J.-P.; Nozha Boujemaa, Conditionally 
Positive Definite Kernels for SVM Based Image Recognition, 
Multimedia and Expo, 2005. ICME 2005. IEEE International Conference 
on , vol., no., pp.113,116, 6-6 July 2005


### Log Kernel

The log kernel is a conditionally positive definite kernel. The function
is given by:

    k(x,y) = -log(α‖x-y‖²ᵞ + 1)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, γ ∈ (0,1]

---
Boughorbel, S.; Tarel, J.-P.; Nozha Boujemaa, Conditionally 
Positive Definite Kernels for SVM Based Image Recognition,
Multimedia and Expo, 2005. ICME 2005. IEEE International Conference 


## Sigmoid Kernel
 
The sigmoid kernel is only positive semidefinite. It is used in the
field of neural networks where it is often used as the activation
function for artificial neurons.

    k(x,y) = tanh(αxᵀy + c)    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0

