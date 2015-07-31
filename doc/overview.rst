Overview
========

Kernels
-------

The kernel methods are a class of algorithms that are used for pattern analysis. These methods make
use of **kernel** functions. A symmetric, real valued kernel function 
:math:`\kappa: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}` is said to be **positive 
definite** or **Mercer** if and only:

.. math::

    \sum_{i=1}^n \sum_{j=1}^n c_i c_j \kappa(\mathbf{x}_i,\mathbf{x}_j) \geq 0

for all :math:`n \in \mathbb{N}`, :math:`\{\mathbf{x}_1, \dots, \mathbf{x}_n\} \subseteq \mathcal{X}`
and :math:`\{c_1, \dots, c_n\} \subseteq \mathbb{R}`. Similarly, a real valued kernel function
is said to be **negative definite** if and only if:

.. math::

    \sum_{i=1}^n \sum_{j=1}^n c_i c_j \kappa(\mathbf{x}_i,\mathbf{x}_j) \leq 0 \qquad \sum_{i=1}^n c_i = 0

for :math:`n \geq 2`, :math:`\{\mathbf{x}_1, \dots, \mathbf{x}_n\} \subseteq \mathcal{X}` and 
:math:`\{c_1, \dots, c_n\} \subseteq \mathbb{R}`. In machine learning literature, **conditionally
positive definite** kernels are often studied instead. This is simply a reversal of the above
inequality. Trivially, every negative definite kernel can be transformed into a conditionally
positive definite kernel by negation.

Several of the most popular kernels have been predefined for quick instantiation as they fall
into a more general class of kernel. For example:

.. code-block:: julia

    GaussianKernel(α)     # Exponentiation of the squared distance multiplied by α
    LaplacianKernel(α)    # Exponentiation of the distance multiplied by α
    RadialBasisKernel(α)  # Identical to GaussianKernel()

    PolynomialKernel(α,c,d)   # Polynomial kernel of degree d
    LinearKernel(α,c)         # Polynomial kernel of degree d = 1

    SigmoidKernel()  # The sigmoid "kernel" (this kernel is neither Mercer or negative definite)

Many other kernels have been predefined. See the section on :ref:`basekernels` and 
:ref:`compositekernels` for a listing of kernels.

To evaluate a kernel, the ``kernel`` function can be used. See the interface for kernel_ function
evaluation.

Kernels may be inspected using the ``ismercer`` and ``isnegdef`` functions to determine if the
kernel is positive or negative definite. See the interface for ismercer_ and isnegdef_ 
respectively.

Both Mercer kernels and negative definite kernels are closed under addition with another kernel
or a positive constant. Addition can be used to generate a new kernel:

.. code-block:: julia

    # Mercer kernel combination
    ScalarProductKernel() + 2.0
    ScalarProductKernel() + MercerSigmoidKernel()
    ScalarProductKernel() + MercerSigmoidKernel() + 2.0

    # Negative definite kernel combination
    SquaredDistanceKernel() + 2.0
    SquaredDistanceKernel() + ChiSquaredKernel()
    SquaredDistanceKernel() + ChiSquaredKernel() + 2.0

Mercer kernels are also closed under multiplication:

.. code-block:: julia

    # Mercer kernel multiplication
    ScalarProductKernel() * 2.0
    ScalarProductKernel() * MercerSigmoidKernel()
    ScalarProductKernel() * MercerSigmoidKernel() * 2.0

Negative definite kernels may only be multiplied by a positive scalar:

.. code-block:: julia

    # Negative definite kernel scaling
    ChiSquaredKernel() * 2


Kernel Matrices
----------------

By default, the input matrices ``X`` and ``Y`` are assumed to be stored in the same format as a
design matrix. In other words, each row of data is assumed to correspond to a vector of variables:

.. math:: \mathbf{X} = \begin{bmatrix} \leftarrow \mathbf{x}_1 \rightarrow  \\ \leftarrow \mathbf{x}_2 \rightarrow   \\ \vdots \\ \leftarrow \mathbf{x}_n \rightarrow \end{bmatrix}
          \qquad
          \mathbf{X}^{\intercal} = \begin{bmatrix} \uparrow & \uparrow & & \uparrow  \\ \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x_n}   \\ \downarrow & \downarrow & & \downarrow \end{bmatrix}

For a single input matrix, the kernel matrix is defined:

.. math:: \mathbf{K}(\mathbf{X}) = \left[\kappa(\mathbf{x}_i,\mathbf{x}_j)\right]_{i,j} \qquad \forall i, j \in \{1, \dots, n\}

For two input matrices:

.. math:: \mathbf{K}(\mathbf{X}, \mathbf{Y}) = \left[\kappa(\mathbf{x}_i,\mathbf{y}_j)\right]_{i,j} \qquad \forall i \in \{1, \dots, n\}, \; j \in \{1, \dots, m\}

See the interface for kernelmatrix_ computation.

Kernel Approximation
--------------------

The **Nystrom method** can be used to approximate squared kernel matrices when full computation becomes
prohibitively expensive. The underlying approximation uses an eigendecomposition. Note that the 
computational complexity of an eigendecomposition is :math:`\mathcal{O}(|s|^3)` where :math:`s`
is the set of sampled vectors. See the interface for nystrom_.


Interface
---------

.. _kernel:

.. function:: kernel(κ::BaseKernel{T}, x::Vector{T}, y::Vector{T})

    Evaluate the kernel of two vectors. Type ``T`` may be any subtype of ``FloatingPoint``.

.. _ismercer:

.. function:: ismercer(::Kernel)

    Returns ``true`` if the kernel type is a Mercer kernel.

.. _isnegdef:

.. function:: isnegdef(::Kernel)

    Returns ``true`` if the kernel type is a negative definite kernel.

.. _kernelmatrix:

.. function:: kernelmatrix(κ::Kernel{T}, X::Matrix{T}; is_trans::Bool, store_upper::Bool, symmetrize::Bool)

    Compute the square kernel matrix of ``X``. Returns kernel matrix ``K``. Type ``T`` may be any
    subtype of ``FloatingPoint``. The following optional arguments may be used positionally or as 
    keyword arguments:

     ``is_trans = false``
       Set ``is_trans = true`` when each column of ``X`` corresponds to a vector of variables.
       Otherwise, each row of ``X`` is treated as a vector of variables.
     ``store_upper = true``
       Set ``store_upper = true`` to compute the upper triangle of the kernel matrix of ``X``. 
       Otherwise, the lower triangle will be computed. This argument will have no impact on the 
       output matrix when ``symmetrize = true``.
     ``symmetrize = true``
       Set ``symmetrize = true`` to copy the contents of the computed triangle to the uncomputed
       triangle.

    If the matrix ``K`` has been pre-allocated, the following method may be used to overwrite 
    ``K`` instead of allocating a new array:

    .. code-block:: julia

        kernelmatrix!(K, κ, X, is_trans, store_upper, symmetrize)


.. function:: kernelmatrix(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)

    Compute the rectangular kernel matrix of ``X`` and ``Y``. Returns kernel matrix ``K``. Type 
    ``T`` may be any subtype of ``FloatingPoint``. The following optional argument may be used 
    positionally or as a keyword argument:

     ``is_trans = false``
       Set ``is_trans = true`` when each column of ``X`` and ``Y`` corresponds to a vector of 
       variables. Otherwise, each row of ``X`` and ``Y`` is treated as a vector of variables.

    If the matrix ``K`` has been pre-allocated, the following method may be used to overwrite 
    ``K`` instead of allocating a new array:

    .. code-block:: julia

        kernelmatrix!(K, κ, X, Y, is_trans)

.. _nystrom:

.. function:: nystrom(κ::Kernel{T}, X::Matrix{T}, s::Array{U}; is_trans::Bool, store_upper::Bool, symmetrize::Bool)

    Compute the Nystrom approximation of the square kernel matrix of ``X``. Returns kernel matrix
    ``K``. Type ``T`` may be any subtype of ``FloatingPoint`` and ``U`` may be any subtype of 
    ``Integer``. The array ``S`` must be a list of observations that have been selected as a 
    sample. The sample may be selected with replacement. The following optional arguments may be 
    used positionally or as keyword arguments:

     ``is_trans = false``
       Set ``is_trans = true`` when each column of ``X`` corresponds to a vector of variables.
       Otherwise, each row of ``X`` is treated as a vector of variables.
     ``store_upper = true``
       Set ``store_upper = true`` to compute the upper triangle of the kernel matrix of ``X``. 
       Otherwise, the lower triangle will be computed. This argument will have no impact on the 
       output matrix when ``symmetrize = true``.
     ``symmetrize = true``
       Set ``symmetrize = true`` to copy the contents of the computed triangle to the uncomputed
       triangle.

    If the matrix ``K`` has been pre-allocated, the following method may be used to overwrite 
    ``K`` instead of allocating a new array:

    .. code-block:: julia

        nystrom!(K, κ, X, s, is_trans, store_upper, symmetrize)


