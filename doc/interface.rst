Standard Interface
==================




Kernel Matrices
----------------

By default, the input matrices ``X`` and ``Y`` are assumed to be stored in the same format as a
design matrix. In other words, each row of data is assumed to correspond to a vector of variables:

.. math:: \mathbf{X} = \begin{bmatrix} \leftarrow \mathbf{x}_1 \rightarrow  \\ \leftarrow \mathbf{x}_2 \rightarrow   \\ \vdots \\ \leftarrow \mathbf{x}_n \rightarrow   \end{bmatrix}

The following methods can be used to compute a kernel matrix where ``T`` may be any subtype of 
``FloatingPoint``.:

.. function:: kernelmatrix(κ::Kernel{T}, X::Matrix{T}; is_trans::Bool, store_upper::Bool, symmetrize::Bool)

    Computes the square kernel matrix ofmatrix ``X``. Returns kernel matrix ``K``. The following 
    optional arguments may be used positionally or as keyword arguments:

     ``is_trans = false`` 
       When true, then each column of ``X`` is treated as a vector of variables for the kernel 
       calculation and resulting kernel matrix will be :math:`m \times m`. Otherwise, each row of 
       ``X`` is assumed to be a vector and the resulting kernel matrix will be :math:`n \times n`.
     ``store_upper = true`` 
       When true, compute the upper half of the kernel matrix of ``X``. Otherwise, the lower 
       half of the kernel matrix is computed. This argument will have no impact on the output
       matrix when ``symmetrize = true``.
     ``symmetrize = true``
       When true, both the upper and lower triangle of the kernel matrix is computed.

.. function:: kernelmatrix!(K::Matrix{T}, κ::Kernel{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool, symmetrize::Bool)

    Computes the square kernel matrix of :math:`n \times m` matrix ``X``. Overwrites ``K`` in 
    the process. Returns kernel matrix ``K``.

.. function:: kernelmatrix(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)

    Computes the kernel matrix of ``X`` and ``Y``. Returns kernel matrix ``K``. The following 
    optional argument may be used positionally or as a keyword argument:

     ``is_trans = false``
       When true:
        - ``X`` must be :math:`p \times n`
        - ``Y`` must be :math:`p \times m`
        - ``K`` will be :math:`n \times m`
       Otherwise:
        - ``X`` must be :math:`n \times p`
        - ``Y`` must be :math:`m \times p`
        - ``K`` will be :math:`n \times m`

.. function:: kernelmatrix(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)

    Computes the kernel matrix of ``X`` and ``Y``. Overwrites matrix ``K`` in the process. Returns
    kernel matrix ``K``. 


Kernel Approximation
--------------------


Kernel Algebra
--------------

.. code-block:: julia

    SineSquaredKernel()   # Sine Squared kernel with t = 1.0
    SineSquaredKernel(t)  # Sine Squared kernel specified t value

