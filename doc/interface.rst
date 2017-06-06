=========
Interface
=========

----------
Essentials
----------

.. function:: ismercer(κ::Kernel) -> Bool

    Returns ``true`` if kernel ``κ`` is a Mercer kernel; ``false`` 
    otherwise.


.. function:: isnegdef(κ::Kernel) -> Bool

    Returns ``true`` if the kernel ``κ`` is a negative definite kernel; 
    ``false`` otherwise.


.. function:: kernel(κ::Kernel, x, y) 

    Apply the kernel ``κ`` to ``x`` and ``y`` where ``x`` and ``y``
    are vectors or scalars of some subtype of ``Real``.


.. function:: centerkernelmatrix(K::Matrix)

    Centers the (rectangular) kernel matrix ``K`` with respect to the implicit
    Kernel Hilbert Space according to the following formula:

    .. math:: [\mathbf{K}]_{ij} = 
        \langle\phi(\mathbf{x}_i) -\mathbf{\mu}_{\phi\mathbf{x}}, 
        \phi(\mathbf{y}_j) - \mathbf{\mu}_{\phi\mathbf{y}} \rangle 
    
    Where :math:`\mathbf{\mu}_{\phi\mathbf{x}}` and 
    :math:`\mathbf{\mu}_{\phi\mathbf{x}}` are given by:

    .. math::

        \mathbf{\mu}_{\phi\mathbf{x}} =  \frac{1}{n} \sum_{i=1}^n \phi(\mathbf{x}_i)
        \qquad \qquad
        \mathbf{\mu}_{\phi\mathbf{y}} =  \frac{1}{m} \sum_{i=1}^m \phi(\mathbf{y}_i)



.. function:: centerkernelmatrix!(K::Matrix)

    The same as ``centerkernelmatrix`` except that ``K`` is overwritten.


---------------
Kernel Matrices
---------------

.. function:: kernelmatrix([σ::MemoryLayout,] κ::Kernel, X::Matrix [, symmetrize::Bool])

    Calculate the kernel matrix of ``X`` with respect to kernel ``κ``. 
    
    See the `format notes`_ to determine the value of ``σ``; by default ``σ`` is
    set to ``RowMajor()``. Set ``symmetrize`` to ``false`` to fill only the 
    upper triangle of ``K``, otherwise the upper triangle will be copied to the
    lower triangle.


.. function:: kernelmatrix!(P::Matrix, σ::MemoryLayout, κ::Kernel, X::Matrix, symmetrize::Bool)

    Identical to ``kernelmatrix`` with the exception that a pre-allocated 
    square matrix ``K`` will be overwritten with the kernel matrix.


.. function:: kernelmatrix([σ::MemoryLayout,] κ::Kernel, X::Matrix, Y::Matrix)

    Calculate the pairwise matrix of ``X`` and ``Y`` with respect to kernel 
    ``κ``. 
    
    See the `format notes`_ to determine the value of ``σ``. By default 
    ``σ`` is set to ``RowMajor``.


.. function:: kernelmatrix!(K::Matrix, σ::MemoryLayout, κ, X::Matrix, Y::Matrix)

    Identical to ``kernelmatrix`` with the exception that a pre-allocated matrix
    ``K`` will be overwritten with the kernel matrix.


.. class:: MemoryLayout()

    The ``MemoryLayout`` abstract type is used to designate which storage layout
    is utilized by a data matrix. There are two concrete subtypes that
    correspond to the two ways of storing a dense matrix:

        * ``RowMajor`` is used to specify that each row of a data matrix 
          corresponds to an observation. 

        * ``ColumnMajor`` is used to specify that each column of a data matrix 
          corresponds to an observation.

    Note that row-major and column-major ordering in this context do not refer
    to the physical storage ordering of the underlying matrices (in the case of
    Julia, all arrays are in column-major ordering). These properties refer to
    the ordering of observations within a data matrix; either per-row or
    per-column. See the `format notes`_ below.


.. _format notes:

.. note::

    Data matrices :math:`X` and :math:`Y` may be stored in one of two formats: 
    row-major ordering or column-major ordering with respect to obversations. 
    Row major ordering is used when each observation vector corresponds to a row
    in the matrix. Conversely,column-major ordering is used when each column 
    corresponds to an observation. For example, for data matrix :math:`X` 
    consisting of observations 
    :math:`\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n`:
    
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

    When row-major ordering is used, then the kernel matrix of :math:`X` will 
    match the dimensions of :math:`X^{\intercal}X`. Otherwise, the kernel matrix
    will match the dimensions of :math:`XX^{\intercal}`. Similarly, the kernel
    matrix will match the dimension of :math:`X^{\intercal}Y` for row-major 
    ordering of :math:`X` and :math:`Y`. Otherwise, the pairwise matrix will 
    match the dimensions of :math:`XY^{\intercal}`.


---------------------------
Kernel Matrix Approximation
---------------------------

.. function:: nystrom(σ::MemoryLayout, κ::Kernel, X::Matrix, S::Vector) -> NystromFact

    Computes a factorization of Nystrom approximation of the square kernel
    matrix of data matrix ``X`` with respect to kernel ``κ``. Returns type
    ``NystromFact`` which stores a Nystrom factorization:

    .. math:: \mathbf{K} \approx \mathbf{C}^{\intercal}\mathbf{WC} 


    .. note::

        The Nystrom method uses an eigendecomposition of the sample of ``X`` to
        estimate ``K``. Generally, the order of ``K`` must be quite large and 
        the sampling ratio small (ex. 15% or less) for the cost of the computing 
        the full kernel matrix to exceed that of the eigendecomposition. This
        method will be more effective for kernels that are not a direct function
        of the dot product as they are not able to make use of BLAS in computing
        the full ``K`` and the cross-over point will occur for smaller ``K``.

.. function:: kernelmatrix(CtWC::NystromFact])

    Computes the approximate kernel matrix using a Nystrom factorization.

----------------
Hyper Parameters
----------------

Behind the scenes, each ``Kernel`` is a collection of ``HyperParameter`` values. 
The hyper parameter type stores the current value of the hyper parameter as well
as an ``Interval`` type that restricts the values that a hyper parameter may
take. The interval type only supports box constraints, but this is sufficient
for the hyper parameters for kernels.

Often, hyper parameter values are restricted to an interval with an open bounded
startpoint or endpoint (ex. :math:`\gamma > 0`). Exclusive endpoints such as
these are often disallowed in optimization algorithms. Therefore, this module
includes a method that is used to apply a transformation, :math:`\theta`, to the
hyper parameter that converts open bounded endpoints into closed or unbounded
endpoints.

 * If :math:`a < \gamma < b` for finite :math:`a` and :math:`b`:

    .. math::

        \theta(\gamma) = \log(\gamma-a) - \log(\gamma-b)
        \qquad \implies \qquad
        \gamma(\theta) = \frac{b\exp(\theta)+a}{1+\exp(\theta)}

 * If :math:`a < \gamma` for finite :math:`a`:

    .. math::

        \theta(\gamma) = \log(\gamma-a)
        \qquad \implies \qquad
        \gamma(\theta) = \exp(\theta)+a


.. function:: ClosedBound(a::Real) -> ClosedBound

    Constructs a ``ClosedBound`` type which is used to signify a closed bound 
    on an interval.

.. function:: OpenBound(a::Real) -> OpenBound

    Constructs an ``OpenBound`` type which is used to signify an open bound on 
    an interval. Type ``T`` must not be integer - only closed bounds are used 
    for integers.

.. function:: NullBound(a::DataType) -> NullBound

    Constructs a ``NullBound`` type which is used to signify an infinite open 
    bound on an interval.

.. function:: Interval(a::Bound,b::Bound) -> Interval

    Constructs an ``Interval`` type. The interval type is used to represent box
    constraints on parameters. This can be used to restrict the values a hyper
    parameter may take on. The ends of the interval may be open (and
    potentially unbounded) or closed.

    The ``Interval`` type is also used to define a transformation, ``theta``
    (not exported) that will transform a parameter restricted to an open bounded
    interval to a parameter restricted a closed or unbounded interval.

    The interval type is used to 

