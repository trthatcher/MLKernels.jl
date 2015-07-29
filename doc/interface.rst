Interface
=========




Kernel Matrices
----------------

By default, the input matrices ``X`` and ``Y`` are assumed to be stored in the same format as a
design matrix. In other words, each row of data is assumed to correspond to a vector of variables:

.. math:: \mathbf{X} = \begin{bmatrix} \leftarrow \mathbf{x}_1 \rightarrow  \\ \leftarrow \mathbf{x}_2 \rightarrow   \\ \vdots \\ \leftarrow \mathbf{x}_n \rightarrow   \end{bmatrix}


.. function:: kernelmatrix(κ::Kernel{T}, X::Matrix{T}; is_trans::Bool, store_upper::Bool, symmetrize::Bool)

    Compute the squared kernel matrix of ``X``. Returns kernel matrix ``K``. Type ``T`` may be any
    subtype of ``FloatingPoint``.The following arguments are required:
    
     ``κ``
       Kernel to apply to each pair of variable vectors in ``X``. 
     ``X``
       The set of variable vectors. Argument ``is_trans`` specifies how to interpret ``X``.
    
    The following optional arguments may be used positionally or as keyword arguments:

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
    ``T`` may be any subtype of ``FloatingPoint``. The following arguments are required:
    
     ``κ``
       Kernel to apply to each pair of variable vectors in ``X``. 
     ``X``
       A set of variable vectors. Argument ``is_trans`` specifies how to interpret ``X``.
     ``Y``
       A set of variable vectors. Argument ``is_trans`` specifies how to interpret ``Y``.

    The following optional argument may be used positionally or as a keyword argument:

     ``is_trans = false``
       Set ``is_trans = true`` when each column of ``X`` and ``Y`` corresponds to a vector of 
       variables. Otherwise, each row of ``X`` and ``Y`` is treated as a vector of variables.

    If the matrix ``K`` has been pre-allocated, the following method may be used to overwrite 
    ``K`` instead of allocating a new array:

    .. code-block:: julia

        kernelmatrix!(K, κ, X, Y, is_trans)

Kernel Approximation
--------------------


Kernel Algebra
--------------

.. code-block:: julia

    SineSquaredKernel()   # Sine Squared kernel with t = 1.0
    SineSquaredKernel(t)  # Sine Squared kernel specified t value

