.. _pageinterface:

Interface
=========

.. _kernel:

.. function:: kernel(κ::BaseKernel{T}, x::Vector{T}, y::Vector{T})

    Evaluate the kernel of two vectors. Type ``T`` may be any subtype of ``FloatingPoint``.

.. _ismercer:

.. function:: ismercer(::Kernel)

    Returns ``true`` if the kernel type is a Mercer kernel.

.. _isnegdef:

.. function:: isnegdef(::Kernel)

    Returns ``true`` if the kernel type is a negative definite kernel.

.. _attainszero:

.. function:: attainszero(::Kernel)

    Returns ``true`` if the kernel can attain zero over its domain.

.. _isnonnegative:

.. function:: isnonnegative(::Kernel)

    Returns ``true`` if the kernel is greater than or equal to zero over its
    domain.

.. _ispositive:

.. function:: ispositive(::Kernel)

    Returns ``true`` if the kernel is greater than zero over its domain.

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

.. _center_kernelmatrix:

.. function:: centerkernelmatrix(X::Matrix{T})

    Centers an ``n`` by ``n`` kernel matrix ``K`` according to the following formula:

    .. math:: \mathbf{K}_{ij} = (\phi(\mathbf{x}_i) -\mathbf{\mu}_\phi)^{\intercal} (\phi(\mathbf{x}_j) - \mathbf{\mu}_\phi) \qquad \text{where} \quad \mathbf{\mu}_\phi =  \frac{1}{n} \sum_{i=1}^n \phi(\mathbf{x}_i)

    If the matrix ``K`` has been pre-allocated, the following method may be used to overwrite 
    ``K`` instead of allocating a new array:

    .. code-block:: julia

        centerkernelmatrix!(K)

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
