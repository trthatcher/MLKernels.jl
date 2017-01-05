=========
Interface
=========

------------
Installation
------------

The package may be added by running one of the following lines of code:

.. code-block:: julia

    # Latest stable release in Metadata:
    Pkg.add("MLKernels")

    # Most up-to-date:
    Pkg.checkout("MLKernels")

    # Development:
    Pkg.checkout("MLKernels", "dev")

----------------
Kernel Kernels
----------------

Kernel functions are implemented as a subtype of ``RealKernel``, a generic
type to represent scalar, real-valued functions such as Mercer kernels or metric
functions. The ``RealKernel`` type has three subtypes: 

  ``PairwiseKernel``
      A type to represent real-valued functions of the form :math:`\mathbb{R}^n 
      \times \mathbb{R}^n \rightarrow \mathbb{R}`.
  ``CompositeKernel``
      A type used to represent real-valued functions of the form :math:`g \circ
      f` where :math:`f:\mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}`
      and :math:`g:\mathbb{R} \rightarrow \mathbb{R}`. This type is constructed
      useing a ``PairwiseKernel`` type for :math:`f` and a
      ``CompositionClass`` type for :math:`g`.
  ``PointwiseKernel``
      A type used to represent scalar transformations of the form :math:`a \cdot
      f(x) + c` for :math:`a \in \mathbb{R}_{++}` and :math:`c \in 
      \mathbb{R}_{+}`, as well as function products and sums of the form 
      :math:`f(x) \cdot g(x)` and :math:`f(x) + g(x)`.


..................
Pairwise Kernels
..................

The following ``PairwiseKernel`` types have been pre-defined:

========================== ==========================
Pairwise Kernel          Constructor
========================== ==========================
:ref:`Euclidean`           ``Euclidean()``
:ref:`Squared Euclidean`   ``SquaredEuclidean()``
:ref:`Chi Squared`         ``ChiSquared()``
:ref:`Scalar Product`      ``ScalarProduct(t)``
:ref:`Sine Squared Kernel` ``SineSquaredKernel(p=π)``
========================== ==========================

...................
Composite Kernels
...................

A number of pre-defined composite function kernels are defined below:

================================ ====================================
Kernel                           Constructor
================================ ====================================
:ref:`Gaussian Kernel`           ``GaussianKernel(α=1)`` 
:ref:`Laplacian Kernel`          ``LaplacianKernel(α=1)``
:ref:`Periodic Kernel`           ``PeriodicKernel(α=1,p=π)``
:ref:`Rational Quadratic Kernel` ``RationalQuadraticKernel(α=1,β=1)`` 
:ref:`Matern Kernel`             ``MaternKernel(ν=1,θ=1)``
:ref:`Polynomial Kernel`         ``PolynomialKernel(a=1,c=1,d=3)``
:ref:`Sigmoid Kernel`            ``SigmoidKernel(α=1,c=1)``
================================ ====================================

Additional kernels can be constructed using the ``CompositeKernel`` type:

.. function:: CompositeKernel(g, f)

    Constructs a ``CompositeKernel`` type. Argument ``g`` must be a 
    ``CompositionClass``. Argument ``f`` must be a ``PairwiseKernel`` that can
    be composed with ``g``.

    The binary operator ``∘`` (``\circ`` in the terminal) is shorthand for this
    constructor. The code block below illustrates how to manually create the
    Gaussian kernel:

    .. code-block:: julia

        α = 1.0
        g = ExponentialClass(α)
        f = Euclidean()

        CompositeKernel(g,f) == (g ∘ f)

    Below is a listing of pre-defined ``CompositionClass`` types that may be
    combined with the ``PairwiseKernel`` types listed above:

    ============================== =====================================
    Composition Class              Constructor
    ============================== =====================================
    :ref:`Exponential Class`       ``ExponentialClass(α=1)``
    :ref:`Gamma Exponential Class` ``GammaExponentialClass(α=1,γ=0.5)``
    :ref:`Rational Class`          ``RationalClass(α=1,β=1)``
    :ref:`Gamma Rational Class`    ``GammaRationalClass(α=1,γ=0.5,β=1)``
    :ref:`Matern Class`            ``MaternClass(ν=1,ρ=1)``
    :ref:`Exponentiated Class`     ``ExponentiatedClass(a=1,c=1)``
    :ref:`Polynomial Class`        ``PolynomialClass(a=1,c=0,d=3)``
    :ref:`Power Class`             ``PowerClass(a=1,c=1,γ=0.5)``
    :ref:`Log Class`               ``LogClass(α=1)``
    :ref:`Gamma Log Class`         ``GammaLogClass(α=1,γ=0.5)``
    :ref:`Sigmoid Class`           ``SigmoidClass(a=1,c=1)``
    ============================== =====================================

...................
Pointwise Kernels
...................

.. function:: AffineKernel(a, c, f)

    Constructs an ``AffineKernel`` type. Argument ``a`` must be a positive
    variable. Argument ``c`` must be a non-negative variable. Argument ``f``
    must be a ``RealKernel``.

    The ``AffineKernel`` will be constructed from arithmetic between a
    ``RealKernel`` type and a ``Real`` type:

    .. code-block:: julia

        a = 2.0
        c = 3.0
        f = ChiSquared()

        AffineKernel(a,c,f) == a*f + c


.. function:: KernelSum(g, f)

    Constructs an ``KernelSum`` type. Argument ``g`` must be a 
    ``RealKernel``. Argument ``f`` must be a ``RealKernel``.

    The ``KernelSum`` will be constructed from arithmetic between two
    ``RealKernel`` types:

    .. code-block:: julia

        g = Euclidean()
        f = ChiSquared()

        KernelSum(g,f) == g + f


.. function:: KernelProduct(g, f)

    Constructs an ``KernelProduct`` type. Argument ``g`` must be a 
    ``RealKernel``. Argument ``f`` must be a ``RealKernel``.

    The ``KernelProduct`` will be constructed from arithmetic between two
    ``RealKernel`` types:

    .. code-block:: julia

        g = Euclidean()
        f = ChiSquared()

        KernelProduct(g,f) == g * f


-------------------------
Kernel Matrix Calculation
-------------------------

A generic ``pairwise`` and ``pairwisematrix`` function is given for computation
of ``RealKernel`` pairwise matrices. If a function is a kernel, then the
corresponding pairwise matrix would be referred to as a kernel matrix.
Similarly, if the real function is a metric, then the pairwise matrix would be a
distance matrix.

.. function:: pairwise(f, x, y) 

    Apply the ``RealKernel`` ``f`` to ``x`` and ``y`` where ``x`` and ``y``
    are vectors or scalars of some subtype of ``Real``.

    This function may also be called using ``kernel`` instead.

.. function:: pairwisematrix([σ,] f, X [, symmetrize])

    Calculate the pairwise matrix of ``X`` with respect to ``RealKernel``
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
    ``RealKernel`` ``f``.

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

