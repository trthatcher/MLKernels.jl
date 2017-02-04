=========
Interface
=========

.. _format notes:

.. note::

    By default, the input matrices ``X`` and ``Y`` are assumed to be stored in 
    the same format as a data matrix (or design matrix) in multivariate 
    statsitics. In other words, each row of ``X`` and ``Y`` is assumed to
    correspond to an observation vector:

    .. math:: \mathbf{X}_{row} = 
                  \begin{bmatrix} 
                      \leftarrow \mathbf{x}_1 \rightarrow \\ 
                      \leftarrow \mathbf{x}_2 \rightarrow \\ 
                      \vdots \\ 
                      \leftarrow \mathbf{x}_n \rightarrow 
                   \end{bmatrix}
              \qquad
              \mathbf{X}_{col} = \mathbf{X}_{row}^{\intercal} = 
                  \begin{bmatrix}
                      \uparrow & \uparrow & & \uparrow  \\
                      \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x_n} \\
                      \downarrow & \downarrow & & \downarrow
                  \end{bmatrix}

    Two subtypes of ``MemoryLayout`` have been created to specify whether a data
    matrix is stored in row-major or column-major ordering:

      * ``RowMajor`` is used to specify that each row of a data matrix 
        corresponds to an observation. 
      * ``ColumnMajor`` is used to specify that each column of a data matrix 
        corresponds to an observation.
               
    When row-major ordering is used, then the kernel matrix of ``X`` will match
    the dimensions of `X'X``. Otherwise, the kernel matrix will match the 
    dimensions of ``X * X'``. Similarly, the kernel matrix will match the 
    dimension of ``X'Y`` for row-major ordering of ``X`` and ``Y``. Otherwise, 
    the pairwise matrix will match the dimensions of ``X * Y'``.


.. function:: ismercer(κ::Kernel) -> Bool

    Returns ``true`` if kernel ``κ`` is a Mercer kernel; ``false`` 
    otherwise.


.. function:: isnegdef(κ::Kernel) -> Bool

    Returns ``true`` if the kernel ``κ`` is a negative definite kernel; 
    ``false`` otherwise.


.. function:: kernel(κ::Kernel, x, y) 

    Apply the kernel ``κ`` to ``x`` and ``y`` where ``x`` and ``y``
    are vectors or scalars of some subtype of ``Real``.


.. function:: kernelmatrix([σ::MemoryLayout,] κ::Kernel, X::Matrix [, symmetrize::Bool])

    Calculate the kernel matrix of ``X`` with respect to kernel ``κ``. The 
    following arguments can be used:
    
      * ``σ`` - see the `format notes`_ to determine the value of ``σ``
      * ``symmetrize`` - set to ``false`` to fill only the upper triangle of 
        ``K``, otherwise the upper triangle will be copied to the lower triangle


.. function:: kernelmatrix!(P::Matrix, σ::MemoryLayout, κ::Kernel, X::Matrix, symmetrize::Bool)

    Identical to ``kernelmatrix`` with the exception that a pre-allocated 
    square matrix ``K`` will be overwritten with the kernel matrix.


.. function:: kernelmatrix([σ::MemoryLayout,] κ::Kernel, X::Matrix, Y::Matrix)

    Calculate the pairwise matrix of ``X`` and ``Y`` with respect to kernel 
    ``κ``. See the `format notes`_ to determine the value of ``σ``. By default 
    ``σ`` is set to ``RowMajor``.


.. function:: kernelmatrix!(K::Matrix, σ::MemoryLayout, κ, X::Matrix, Y::Matrix)

    Identical to ``kernelmatrix`` with the exception that a pre-allocated matrix
    ``K`` will be overwritten with the kernel matrix.


.. function:: centerkernelmatrix(K::Matrix)

    Centers the square kernel matrix ``K`` with respect to the implicit Kernel 
    Hilbert Space according to the following formula:

    .. math:: [\mathbf{K}]_{ij} = 
        \langle\phi(\mathbf{x}_i) -\mathbf{\mu}_\phi, 
        \phi(\mathbf{x}_j) - \mathbf{\mu}_\phi \rangle 
        \qquad \text{where} \quad 
        \mathbf{\mu}_\phi =  \frac{1}{n} \sum_{i=1}^n \phi(\mathbf{x}_i)

.. function:: centerkernelmatrix!(K::Matrix)

    The same as ``centerkernelmatrix`` except that ``K`` is overwritten.


.. function:: nystrom!(K, κ, X, s, is_trans, store_upper, symmetrize)

    Overwrite the pre-allocated square matrix ``K`` with the Nystrom 
    approximation of the kernel matrix of ``X``. Returns matrix ``K``. Type 
    ``T`` may be any  subtype of ``AbstractFloat`` and ``U`` may be any subtype 
    of ``Integer``. The array ``S`` must be a 1-indexed sample of the 
    observations of ``X`` (with replacement). When ``is_trans`` is set to 
    ``true``, then ``K`` must match the dimensions of ``X'X`` and ``S`` must 
    sample the columns of ``X``. Otherwise, ``K`` must match the dimensions of 
    ``X * X'`` and ``S`` must sample the rows of ``X``.

    Set ``store_upper`` to ``true`` to compute the upper triangle of the kernel 
    matrix of ``X`` or ``false`` to compute the lower triangle. If
    ``symmetrize`` is set to ``false``, then only the specified triangle will be
    computed.

    .. note::

        The Nystrom method uses an eigendecomposition of the sample of ``X`` to
        estimate ``K``. Generally, the order of ``K`` must be quite large and 
        the sampling ratio small (ex. 15% or less) for the cost of the computing 
        the full kernel matrix to exceed that of the eigendecomposition. This
        method will be more effective for kernels that are not a direct function
        of the dot product (Chi-Squared, Sine-Squared, etc.) as they are not
        able to make use of BLAS in computing the full ``K`` and the cross-over
        point will occur for smaller ``K``.

.. function:: nystrom(κ, X, s, [; is_trans, store_upper, symmetrize])

    The same as ``nystrom!`` with matrix ``K`` automatically allocated.
