------------------
Exported Functions
------------------

Below are the functions exported by the ``MLKernels`` module. The package may be
added by running the following code:

.. code-block:: julia

    Pkg.add("MLKernels")


Kernel Properties
-----------------

Some kernel methods are only suited only to Mercer kernels, although other
methods such as Kernel Support Vector Machines and Kernel Principal Components
Analysis do not have that constraint. Similarly, many kernels are defined as a
composition of another kernel. The composed kernel can often be mixed and
matched subject to some restrictions on the range (see :ref:`kernelclasses`). 
The following functions can be used to check for certain conditions:

.. _ismercer:

.. function:: ismercer(κ)

    Returns ``true`` if the ``κ`` kernel is a Mercer kernel; ``false`` 
    otherwise.

.. _isnegdef:

.. function:: isnegdef(κ)

    Returns ``true`` if the kernel ``κ`` is a negative definite kernel; 
    ``false`` otherwise.

.. _isnonnegative:

.. function:: isnonnegative(κ)

    Returns ``true`` if the kernel ``κ`` is *always* greater than or equal to 
    zero over its domain and parameter space; ``false`` otherwise.

.. _ispositive:

.. function:: ispositive(κ)

    Returns ``true`` if the kernel ``κ`` is *always* greater than zero over its
    domain and parameter space; ``false`` otherwise.

Kernel Arithmetic
-----------------

Kernels support a subset of standard arithmetic. See the descriptions below for
restrictions on arithmetic:

=========== =
Operation   Description
=========== =
``κ + ψ``   Addition; see :ref:`kernelaffinity` and :ref:`kernelsum`
``κ * ψ``   Multiplication; see :ref:`kernelaffinity` and :ref:`kernelproduct`
``ϕ ∘ κ``   Function composition; see :ref:`kernelclasses` for combinations.
``exp(κ)``  Exponentiation;
``κ^t``     Power;
``tanh(κ)`` Hyperbolic tangent;
=========== =


Kernel Computation
------------------

.. note::

    By default, the input matrices ``X`` and ``Y`` are assumed to be stored in 
    the same format as a data matrix (or design matrix) in multivariate 
    statsitics. In other words, each row of ``X`` and ``Y`` is assumed to
    correspond to an observation vector:

    .. math:: \mathbf{X} = 
                  \begin{bmatrix} 
                      \leftarrow \mathbf{x}_1 \rightarrow \\ 
                      \leftarrow \mathbf{x}_2 \rightarrow \\ 
                      \vdots \\ 
                      \leftarrow \mathbf{x}_n \rightarrow 
                   \end{bmatrix}
              \qquad
              \mathbf{X}^{\intercal} = 
                  \begin{bmatrix}
                      \uparrow & \uparrow & & \uparrow  \\
                      \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x_n} \\
                      \downarrow & \downarrow & & \downarrow
                  \end{bmatrix}

    In Machine Learning literature, the data matrix is often transposed. The 
    ``is_trans`` for kernel matrix functions can be set to ``false`` to indicate
    that the input matrices are in the transposed format.

.. _kernel:

.. function:: kernel(κ, x, y) 

    Evaluate the kernel function ``κ`` where ``x`` and ``y`` are vectors or 
    scalars of some subtype of ``AbstractFloat``.

.. _kernelmatrix:

.. function:: kernelmatrix!(K, κ, X, is_trans, store_upper, symmetrize)

    Overwrite the pre-allocated square matrix ``K`` with the kernel matrix of 
    ``X`` for kernel ``κ``. When ``is_trans`` is set to ``true``, then ``K`` 
    must match the dimensions of ``X'X``. Otherwise, ``K`` must match the
    dimensions of ``X * X'``.
    
    Set ``store_upper`` to ``true`` to compute the upper triangle of the kernel 
    matrix of ``X`` or ``false`` to compute the lower triangle. If
    ``symmetrize`` is set to ``false``, then only the specified triangle will be
    computed.

.. function:: kernelmatrix(κ, X [; is_trans, store_upper, symmetrize])

    Same as ``kernelmatrix!`` with matrix ``K`` automatically allocated.

.. function:: kernelmatrix!(K, κ, X, Y, is_trans)

    Overwrite the pre-allocated square matrix ``K`` with the kernel matrix of 
    ``X`` and ``Y`` for kernel ``κ``. When ``is_trans`` is set to ``true``, then
    ``K`` must match the dimensions of ``X'Y``. Otherwise, ``K`` must match the
    dimensions of ``X * Y'``.

.. function:: kernelmatrix(κ, X, Y [; is_trans, store_upper, symmetrize])

    Same as ``kernelmatrix!`` with matrix ``K`` automatically allocated.


.. _center_kernelmatrix:

.. function:: centerkernelmatrix!(X)

    In-place centering of square kernel matrix ``K`` in the implicit Kernel
    Hilbert Space according to the following formula:

    .. math:: [\mathbf{K}]_{ij} = 
        \langle\phi(\mathbf{x}_i) -\mathbf{\mu}_\phi, 
        \phi(\mathbf{x}_j) - \mathbf{\mu}_\phi \rangle 
        \qquad \text{where} \quad 
        \mathbf{\mu}_\phi =  \frac{1}{n} \sum_{i=1}^n \phi(\mathbf{x}_i)

.. function:: centerkernelmatrix(X)

    Same as ``centerkernelmatrix!`` but makes a copy of ``X``.

Kernel Approximation
--------------------

.. _nystrom:

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
