Interface
=========

Kernels
-------

Many kernels have been predefined. See the section on :ref:`basekernels` and 
:ref:`compositekernels` for a listing of kernels.

Kernels may be inspected using the ``ismercer`` and ``isnegdef`` functions to determine if the
kernel is positive or negative definite, respectively:

.. function:: ismercer(::Kernel)

    Returns ``true`` if the kernel type is a Mercer kernel.

.. function:: isnegdef(::Kernel)

    Returns ``true`` if the kernel type is a negative definite kernel.


Kernel Algebra
--------------

Both Mercer kernels and negative definite kernels are closed under addition with another kernel
or a positive constant. Addition can be used to generate a new kernel:

.. code-block:: julia

    julia> ScalarProductKernel() + 2.0
    KernelSum{Float64}(2.0, ScalarProductKernel())

    julia> ScalarProductKernel() + MercerSigmoidKernel()
    KernelSum{Float64}(0.0, ScalarProductKernel(), MercerSigmoidProduct(d=0.0,b=1.0))

    julia> ScalarProductKernel() + MercerSigmoidKernel() + 2.0
    KernelSum{Float64}(2.0, ScalarProductKernel(), MercerSigmoidProduct(d=0.0,b=1.0))

    julia> SquaredDistanceKernel() + 2.0
    KernelSum{Float64}(2.0, SquaredDistanceKernel(t=1.0))

    julia> SquaredDistanceKernel() + ChiSquaredKernel()
    KernelSum{Float64}(0.0, SquaredDistanceKernel(t=1.0), ChiSquaredKernel(t=1.0))

    julia> SquaredDistanceKernel() + ChiSquaredKernel() + 2.0
    KernelSum{Float64}(2.0, SquaredDistanceKernel(t=1.0), ChiSquaredKernel(t=1.0))

Mercer kernels are also closed under multiplication:

.. code-block:: julia

    julia> ScalarProductKernel() * 2.0
    KernelProduct{Float64}(2.0, ScalarProductKernel())

    julia> ScalarProductKernel() * MercerSigmoidKernel()
    KernelProduct{Float64}(1.0, ScalarProductKernel(), MercerSigmoidProduct(d=0.0,b=1.0))

    julia> ScalarProductKernel() * MercerSigmoidKernel() * 2.0
    KernelProduct{Float64}(2.0, ScalarProductKernel(), MercerSigmoidProduct(d=0.0,b=1.0))

Negative definite kernels may be multiplied by a positive scalar:

.. code-block:: julia

    julia> ChiSquaredKernel() * 2
    KernelProduct{Float64}(2.0, ChiSquaredKernel(t=1.0))


Kernel Functions
----------------

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

Kernel Approximation
--------------------

The Nystrom method can be used to approximate squared kernel matrices when full computation becomes
prohibitively expensive. The underlying approximation uses an eigen decomposition. Note that the 
computational complexity of an eigen decomposition is :math:`\mathcal{O}(|s|^3)` where :math:`s`
is the set of sampled vectors.

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
