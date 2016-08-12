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

----------------
Kernel Functions
----------------

Kernel functions are implemented as a subtype of ``RealFunction``, a generic
type to represent scalar, real-valued functions such as Mercer kernels or metric
functions. A list of pre-defined kernels is avaiable here.

A number of constructors are provided to construct kernels and functions beyond
the constructors provided above. See the `properties & constructors`_ section 
for additional constructors and for properties of ``RealFunction`` subtypes.

-------------------------
Kernel Matrix Calculation
-------------------------

A generic ``pairwise`` and ``pairwisematrix`` function is given for computation
of ``RealFunction`` pairwise matrices. If a function is a kernel, then the
corresponding pairwise matrix would be referred to as a kernel matrix.
Similarly, if the real function is a metric, then the pairwise matrix would be a
distance matrix.

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


.. function:: centerkernel!(K)

    In-place centering of square kernel matrix ``K`` in the implicit Kernel
    Hilbert Space according to the following formula:

    .. math:: [\mathbf{K}]_{ij} = 
        \langle\phi(\mathbf{x}_i) -\mathbf{\mu}_\phi, 
        \phi(\mathbf{x}_j) - \mathbf{\mu}_\phi \rangle 
        \qquad \text{where} \quad 
        \mathbf{\mu}_\phi =  \frac{1}{n} \sum_{i=1}^n \phi(\mathbf{x}_i)

.. function:: centerkernel(K)

    The same as ``centerkernel!`` except that ``K`` is not overwritten.

.. function:: KernelCenterer(K)

    Gathers the required statistics to center with respect to kernel matrix 
    ``K``. This type can be passed to ``centerkernel!`` or ``centerkernel`` to
    center with respect to these statistics:

    .. code-block:: julia

        κ = GaussianKernel())
        X = rand(30,5)
        Y = rand(20,5)

        Kxx = kernelmatrix(κ, X)     
        Kxy = kernelmatrix(κ, X, Y)

        KC = KernelCenterer(Kxx)

        centerkernel(KC, Kxx)  # By centering w.r.t. X, the left matrix must be
        centerkernel(KC, Kxy)  # X in the kernelmatrix(κ, X, ...) calculation

    The following centering function is used to center with respect to the
    centering statistics:

    .. math:: [\mathbf{K}]_{ij} = 
        \langle\phi(\mathbf{x}_i) -\mathbf{\mu}_{\phi}, 
        \phi(\mathbf{y}_j) - \mathbf{\mu}_\phi \rangle 
        \qquad \text{where} \quad 
        \mathbf{\mu}_\phi =  \frac{1}{n} \sum_{i=1}^n \phi(\mathbf{x}_i)

.. function:: KernelTransformer([σ,] κ, X [, center_kernel, copy_data])

    Constructs a ``KernelTransformer`` type that can be used to compute kernel
    matrices with respect to kernel ``κ`` and data matrix ``X`` (with memory
    ordering ``σ``).
    
    By default, the kernel matrix will be centered with respect to ``X``. The 
    argument ``center_kernel`` can be set to ``false`` to disable centering of 
    the kernel matrix.
    
    Setting ``copy_data`` to ``false`` will prevent a deep copy of the matrix 
    ``X``. However, if ``X`` is modified, then the centering statistics may no
    longer be valid.


---------------------------
Kernel Matrix Approximation
---------------------------

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

.. _`properties & constructors`:

-------------------------
Properties & Constructors
-------------------------

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

.. function:: CompositeFunction(g, f)

    Constructs a ``CompositeFunction`` type. Argument ``g`` must be a 
    ``CompositionClass``. Argument ``f`` must be a ``PairwiseFunction`` that can
    be composed with ``g``.

    The binary operator ``∘`` (``\circ`` in the terminal) is shorthand for this
    constructor. The code block below illustrates how to manually create the
    Gaussian kernel:

    .. code-block:: julia

        α = 1.0
        g = ExponentialClass(α)
        f = Euclidean()

        CompositeFunction(g,f) == (g ∘ f)

    A list of pre-defined composition classes is available here.

.. function:: AffineFunction(a, c, f)

    Constructs an ``AffineFunction`` type. Argument ``a`` must be a positive
    variable. Argument ``c`` must be a non-negative variable. Argument ``f``
    must be a ``RealFunction``.

    The ``AffineFunction`` will be constructed from arithmetic between a
    ``RealFunction`` type and a ``Real`` type:

    .. code-block:: julia

        a = 2.0
        c = 3.0
        f = ChiSquared()

        AffineFunction(a,c,f) == a*f + c


.. function:: FunctionSum(g, f)

    Constructs an ``FunctionSum`` type. Argument ``g`` must be a 
    ``RealFunction``. Argument ``f`` must be a ``RealFunction``.

    The ``FunctionSum`` will be constructed from arithmetic between two
    ``RealFunction`` types:

    .. code-block:: julia

        g = Euclidean()
        f = ChiSquared()

        FunctionSum(g,f) == g + f


.. function:: FunctionProduct(g, f)

    Constructs an ``FunctionProduct`` type. Argument ``g`` must be a 
    ``RealFunction``. Argument ``f`` must be a ``RealFunction``.

    The ``FunctionProduct`` will be constructed from arithmetic between two
    ``RealFunction`` types:

    .. code-block:: julia

        g = Euclidean()
        f = ChiSquared()

        FunctionProduct(g,f) == g * f

-----
Notes
-----

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

