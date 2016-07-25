=========
Interface
=========

------------
Installation
------------

The package may be
added by running one of the following lines of code:

.. code-block:: julia

    # Latest stable release
    Pkg.add("MLKernels")

    # Most up to date
    Pkg.checkout("MLKernels")

    # Development:
    Pkg.checkout("MLKernels", "dev")

----------
Properties
----------

.. _ismercer:

.. function:: ismercer(f)

    Returns ``true`` if the ``f`` function is a Mercer kernel; ``false`` 
    otherwise.

.. _isnegdef:

.. function:: isnegdef(f)

    Returns ``true`` if the funtion ``f`` is a negative definite kernel; 
    ``false`` otherwise.

.. function:: ismetric(f)

    Returns ``true`` if the function ``f`` valid metric; ``false`` otherwise.

.. _isnonnegative:

.. function:: isnonnegative(f)

    Returns ``true`` if the kernel ``f`` is *always* greater than or equal to 
    zero over its domain and parameter space; ``false`` otherwise.

.. _ispositive:

.. function:: ispositive(f)

    Returns ``true`` if the kernel ``f`` is *always* greater than zero over its
    domain and parameter space; ``false`` otherwise.

-------------------
Pre-Defined Kernels
-------------------

A list of pre-defined kernels is available here

-------------------
Kernel Constructors
-------------------

The following operators are short hand for the constructors outlined above:

=========== =============================
Operation   Constructor
=========== =============================
``g + f``   ``FunctionProduct(g, f)``
``g * f``   ``FunctionSum(g, f)``
``g ∘ f``   ``CompositeFunction(g, f)``
=========== =============================

.. function:: CompositeFunction(g, f), g ∘ f

    Constructs a ``CompositeFunction`` type from ``CompositionClass`` ``g`` and
    ``PairwiseFunction`` ``f`` when ``g`` and ``f`` can be composed.

    A list of pre-defined composition classes is available here.

    .. code-block:: julia

        g = ExponentialClass()
        f = Euclidean()

        h = CompositeFunction(g,f)
        h == (g ∘ f)


.. function:: AffineFunction(a, c, f)

    Constructs an ``AffineFunction`` from positive variable ``a``, non-negative
    variable ``c`` and ``RealFunction`` ``f``.

.. function:: FunctionSum(g, f), g + f

    Constructs an ``FunctionSum`` from ``RealFunction`` ``g`` and
    ``RealFunction`` ``f``. 

.. function:: FunctionProduct(g, f), g * f

    Constructs an ``FunctionSum`` from ``RealFunction`` ``g`` and
    ``RealFunction`` ``f``. 

--------------------
Pairwise Computation
--------------------

.. _format notes:

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

    The memory order parameter, ``σ``, can be set to ``Val{:row}`` to use the
    row-major ordering and ``Val{:col}`` for column-major ordering.
    
    When row-major ordering is used, then the pairwise matrix of ``X`` will 
    match the dimensions of `X'X``. Otherwise, the pairwise matrix will match 
    the dimensions of ``X * X'``.

    For ``X`` and ``Y``, pairwise matrix will match the dimension of ``X'Y`` for
    row-major ordering. Otherwise, the pairwise matrix will match the dimensions
    of ``X * Y'``.

.. function:: pairwise(f, x, y) 

    Apply the ``RealFunction`` ``f`` to ``x`` and ``y`` where ``x`` and ``y``
    are vectors or scalars of some subtype of ``Real``.

    This function may also be called using ``kernel`` instead.

.. function:: pairwisematrix([σ,] f, X [, symmetrize])

    Calculate the pairwise matrix of ``X`` with respect to ``RealFunction``
    ``f``. Set ``symmetrize`` to ``false`` to populate only the upper triangle 
    of the pairwise matrix.

    See the `format notes`_ to determine the value of ``σ``. By default ``σ`` is
    set to ``Val{:row}``.

    This function may also be called using ``kernelmatrix`` instead.

.. function:: pairwisematrix!(P, σ, f, X, symmetrize)

    Identical to ``pairwisematrix`` with the exception that a pre-allocated 
    square matrix ``P`` may be supplied to be overwritten.

    This function may also be called using ``kernelmatrix!`` instead.


.. function:: pairwisematrix([σ,] f, X, Y)

    Calculate the pairwise matrix of ``X`` and ``Y`` with respect to 
    ``RealFunction`` ``f``.

    See the `format notes`_ to determine the value of ``σ``. By default ``σ`` is
    set to ``Val{:row}``.

    This function may also be called using ``kernelmatrix`` instead.

.. function:: pairwisematrix!(P, σ, f, X, Y)

    Identical to ``pairwisematrix`` with the exception that a pre-allocated 
    square matrix ``P`` may be supplied to be overwritten.

    This function may also be called using ``kernelmatrix!`` instead.


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

----------------------
Pairwise Approximation
----------------------

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
