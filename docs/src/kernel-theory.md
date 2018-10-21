# Kernel Theory

## The Kernel Trick

Many machine and statistical learning algorithms, such as support vector machines and 
principal components analysis, are based on **inner products**. These methods can often be 
generalized through use of the **kernel trick** to create anonlinear decision boundary 
without using an explicit mapping to another space. 

The kernel trick makes use of **Mercer kernels** which operate on vectors in the input 
space but can be expressed as inner products in another space. In other words, if 
``\mathcal{X}`` is the input vector space and ``\kappa`` is the Mercer kernel function, 
then for some vector space ``\mathcal{V}`` there exists a function `\phi` such that:

```math
\kappa(x_1, x_2) 
= \left\langle \phi(x_1), \phi(x_2)\right\rangle_{\mathcal{V}}
\qquad x_1, x_2 \in \mathcal{X}
```

In machine learning, the vector space ``\mathcal{X}`` is known as the feature space and the 
function ``\phi`` is known as a feature map. A simple example of a feature map can be shown 
with the Polynomial Kernel:

```math
\kappa(\mathbf{x},\mathbf{y}) = (a\mathbf{x}^\intercal\mathbf{y} + c)^{d}
\qquad \mathbf{x},\mathbf{y} \in \mathbb{R}^n, 
\quad a, c \in \mathbb{R}_+
\quad d \in \mathbb{Z}_+
```

In our example, we will use ``n=2``, ``d=2``, ``a=1`` and ``c=0``. Substituting these 
values in, we get the following kernel function:

```math
\kappa(\mathbf{x},\mathbf{y}) = \left(x_1 y_1 + x_2 y_2\right)^2
= x_1^2 y_1^2 + x_1 x_2 y_1 y_2 + x_2^2 y_2^2
= \phi(\mathbf{x})^\intercal\phi(\mathbf{y})
```

Where the feature map ``\phi : \mathbb{R}^2 \rightarrow \mathbb{R}^3`` is defined by:

```math
\phi(\mathbf{x}) = 
\begin{bmatrix}
    x_1^2 \\
    x_1 x_2 \\
    x_2^2
\end{bmatrix}
```

The advantage of the implicit feature map is that we may transform non-linearly data into 
linearly separable data in the implicit space.


## Kernels

The kernel methods are a class of algorithms that are used for pattern analysis. These 
methods make use of **kernel** functions. A symmetric, real valued kernel function 
``\kappa: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}`` is said to be **positive 
definite** or **Mercer** if and only:

```math
\sum_{i=1}^n \sum_{j=1}^n c_i c_j \kappa(\mathbf{x}_i,\mathbf{x}_j) \geq 0
```

for all ``n \in \mathbb{N}``, ``\{\mathbf{x}_1, \dots, \mathbf{x}_n\} \subseteq \mathcal{X}``
and ``\{c_1, \dots, c_n\} \subseteq \mathbb{R}``. Similarly, a real valued kernel function
is said to be **negative definite** if and only if:

```math
\sum_{i=1}^n \sum_{j=1}^n c_i c_j \kappa(\mathbf{x}_i,\mathbf{x}_j) \leq 0 \qquad \sum_{i=1}^n c_i = 0
```

for ``n \geq 2``, ``\{\mathbf{x}_1, \dots, \mathbf{x}_n\} \subseteq \mathcal{X}`` and 
``\{c_1, \dots, c_n\} \subseteq \mathbb{R}``. In machine learning literature, 
**conditionally positive definite** kernels are often studied instead. This is simply a 
reversal of the above inequality. Trivially, every negative definite kernel can be 
transformed into a conditionally positive definite kernel by negation.


## Further Reading

* Berg C, Christensen JPR, Ressel P. 1984. *Harmonic Analysis on Semigroups*. Springer-Verlag New York. Chapter 3, General Results on Positive and Negative Definite Matrices and Kernels; p. 66-85.
* Bouboulis P. 2014. *Academic Press Library in Signal Processing, Volume 1: Array and Statistical Signal Processing (1st ed.)*. Academic Press. Chapter 17, Online Learning in Reproducing Kernel Hilbert Spaces; p. 883-987.
* Genton M.G. 2002. *Classes of kernels for machine learning: a statistics perspective*. The Journal of Machine Learning Research. Volume 2 (March 2002), 299-312.
* Rasmussen C, Williams CKI. 2005. *Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning)*. The MIT Press. Chapter 4, Covariance Functions; p. 79-104.
