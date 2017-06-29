=========
Interface
=========

.. _storage notes:

-------
Storage
-------

`MLKernels.jl`_ allows for data matrices to be stored in one of two ways with 
respect to the observations based on parameters provided by the user. In order 
to specify the ordering used, a subtype of the ``MemoryLayout`` abstract type 
can be provided as a parameter to any methods taking matrices as a parameter:

.. type:: RowMajor

    Identifies when each observation vector corresponds to a row in the
    data matrix. This is commonly used in the field of statistics in the context
    of `design matrices`_. For example, for data matrix :math:`\mathbf{X}` 
    consisting of observations 
    :math:`\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n` :

    .. math:: \mathbf{X}_{row} = 
                  \begin{bmatrix} 
                      \leftarrow \mathbf{x}_1 \rightarrow \\ 
                      \leftarrow \mathbf{x}_2 \rightarrow \\ 
                      \vdots \\ 
                      \leftarrow \mathbf{x}_n \rightarrow 
                   \end{bmatrix}
    
    When row-major ordering is used, then the kernel matrix of
    :math:`\mathbf{X}` will match the dimensions of 
    :math:`\mathbf{X}^{\intercal}\mathbf{X}`. Similarly, the kernel matrix will 
    match the dimension of :math:`\mathbf{X}^{\intercal}\mathbf{Y}` for row-major 
    ordering of data matrix :math:`\mathbf{X}` and :math:`\mathbf{Y}`. 


.. type:: ColumnMajor

    Identifies when each observation vector corresponds to the column of the 
    data matrix. This is much more common in Machine Learning communities:

    .. math:: \mathbf{X}_{col} = \mathbf{X}_{row}^{\intercal} = 
                  \begin{bmatrix}
                      \uparrow & \uparrow & & \uparrow  \\
                      \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x_n} \\
                      \downarrow & \downarrow & & \downarrow
                  \end{bmatrix}

    With column-major ordering, the kernel matrix will match the dimensions of 
    :math:`\mathbf{XX}^{\intercal}`. Similarly, the kernel matrix of data 
    matrices :math:`\mathbf{X}` and :math:`\mathbf{Y}` match the dimensions of 
    :math:`\mathbf{XY}^{\intercal}`.

.. note::

    Row-major and column-major ordering in this context do not refer to the 
    physical storage ordering of the underlying matrices (in the case of Julia, 
    all arrays are in column-major ordering). These properties refer to the 
    ordering of observations within a data matrix; either per-row or per-column. 


----------
Essentials
----------

The primary feature of the ``MLKernels`` package is the ability to efficiently
compute kernel functions and kernel matrices. The interface is outlined below:

.. function:: ismercer(κ::Kernel) -> Bool

    Returns ``true`` if kernel ``κ`` is a Mercer kernel; ``false`` 
    otherwise.


.. function:: isnegdef(κ::Kernel) -> Bool

    Returns ``true`` if the kernel ``κ`` is a negative definite kernel; 
    ``false`` otherwise.


.. function:: isisotropic(κ::Kernel) -> Bool

    Returns ``true`` if the kernel ``κ`` is an isotropic kernel; ``false``
    otherwise.


.. function:: isstationary(κ::Kernel) -> Bool

    Returns ``true`` if the kernel ``κ`` is a stationary kernel; ``false``
    otherwise.


.. function:: kernel(κ::Kernel, x, y) 

    Apply the kernel ``κ`` to ``x`` and ``y`` where ``x`` and ``y``
    are vectors or scalars of some subtype of ``Real``.


.. function:: kernelmatrix([σ::MemoryLayout,] κ::Kernel, X::Matrix [, symmetrize::Bool])

    Calculate the kernel matrix of ``X`` with respect to kernel ``κ``. 
    
    See the `storage notes`_ to determine the value of ``σ``; by default ``σ`` 
    is set to ``RowMajor()``. Set ``symmetrize`` to ``false`` to fill only the 
    upper triangle of ``K``, otherwise the upper triangle will be copied to the
    lower triangle.


.. function:: kernelmatrix!(P::Matrix, σ::MemoryLayout, κ::Kernel, X::Matrix, symmetrize::Bool)

    In-place version of ``kernelmatrix`` where pre-allocated matrix ``K`` will 
    be overwritten with the kernel matrix.


.. function:: kernelmatrix([σ::MemoryLayout,] κ::Kernel, X::Matrix, Y::Matrix)

    Calculate the pairwise matrix of ``X`` and ``Y`` with respect to kernel 
    ``κ``. 
    
    See the `storage notes`_ to determine the value of ``σ``. By default 
    ``σ`` is set to ``RowMajor()``.


.. function:: kernelmatrix!(K::Matrix, σ::MemoryLayout, κ::Kernel, X::Matrix, Y::Matrix)

    In-place version of ``kernelmatrix`` where pre-allocated matrix ``K`` will 
    be overwritten with the kernel matrix.


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

    In-place version of ``centerkernelmatrix`` overwriting ``K``.


-------------
Approximation
-------------

In many cases, a fast, approximate results is more important than a perfect
result. The Nystrom method can be used to generate a factorization that can be
used to approximate a large, symmetric kernel matrix. Given data matrix
:math:`\mathbf{X} \in \mathbb{R}^{n \times p}` (one observation per row) and 
kernel matrix :math:`\mathbf{K} \in \mathbb{R}^{n \times n}`, the Nystrom method
takes a sample :math:`S` of the observations of :math:`\mathbf{X}` of size 
:math:`s < n` and generates a factorization such that:

.. math:: \mathbf{K} \approx \mathbf{C}^{\intercal}\mathbf{WC}

Where :math:`\mathbf{W}` is the :math:`s \times s` pseudo-inverse of the sample 
kernel matrix based on :math:`S` and :math:`\mathbf{C}` is a :math:`s \times n`
matrix.

The Nystrom method uses an eigendecomposition of the sample kernel matrix of 
:math:`\mathbf{X}` to estimate :math:`\mathbf{K}`. Generally, the order of 
:math:`\mathbf{K}` must be quite large and the sampling ratio small (ex. 15% or 
less) for the cost of the computing the full kernel matrix to exceed that of the
eigendecomposition. This method will be more effective for kernels that are not 
a direct function of the dot product as they are not able to make use of BLAS in
computing the full matrix :math:`\mathbf{K}` and the cross-over point will occur
for smaller :math:`\mathbf{K}`.

`MLKernels.jl`_ implements the Nystrom approximation:

.. type:: NystromFact

    Type for storing a Nystrom factorization. The factorization contains two
    fields: ``W`` and ``C`` as described above.

.. function:: nystrom(σ::MemoryLayout, κ::Kernel, X::Matrix, S::Vector) -> NystromFact

    Computes a factorization of Nystrom approximation of the square kernel
    matrix of data matrix ``X`` with respect to kernel ``κ``. Returns type
    ``NystromFact`` which stores a Nystrom factorization:

.. function:: kernelmatrix(CtWC::NystromFact])

    Computes the approximate kernel matrix using the Nystrom factorization.


------------------
Pairwise Functions
------------------

The ``PairwiseFunctions`` submodule is provided to compute symmetric real-valued
functions (pairwise functions) of the form:

.. math:: f:\mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}
    
Each kernel function has an underlying pairwise function. Similar to kernel 
functions, a pairwise function can be evaluated for each pair of observations 
across two data matrices to produce a matrix of pairwise evaluations. Given 
data matrix :math:`\mathbf{X}`, data matrix :math:`\mathbf{Y}` and pairwise 
function :math:`f`, the pairwise matrix :math:`\mathbf{P}` is defined by:

.. math:: \mathbf{P} = \left[f(\mathbf{x}_i, \mathbf{y}_j)\right]_{ij} \quad
    \; \forall \mathbf{x}_i \in \mathbf{X}, \quad \mathbf{y}_j \in \mathbf{Y}

The interface is outlined below:

.. function:: pairwise(f::PairwiseFunction, x, y) 

    Apply the function ``f`` to ``x`` and ``y`` where ``x`` and ``y``
    are vectors or scalars of some subtype of ``Real``.


.. function:: pairwisematrix([σ::MemoryLayout,] f::Kernel, X::Matrix [, symmetrize::Bool])

    Calculate the pairwise matrix of ``X`` with respect to function ``f``. 
    
    See the `storage notes`_ to determine the value of ``σ``; by default ``σ`` 
    is set to ``RowMajor()``. Set ``symmetrize`` to ``false`` to fill only the 
    upper triangle of ``P``, otherwise the upper triangle will be copied to the
    lower triangle.


.. function:: pairwisematrix!(P::Matrix, σ::MemoryLayout, f::PairwiseFunction, X::Matrix, symmetrize::Bool)

    In-place version of ``pairwisematrix`` where pre-allocated matrix ``P`` will 
    be overwritten with the pairwise matrix.


.. function:: pairwisematrix([σ::MemoryLayout,] f::PairwiseFunction, X::Matrix, Y::Matrix)

    Calculate the pairwise matrix of ``X`` and ``Y`` with respect to function
    ``f``.  
    
    See the `storage notes`_ to determine the value of ``σ``. By default 
    ``σ`` is set to ``RowMajor()``.


.. function:: pairwisematrix!(P::Matrix, σ::MemoryLayout, f::PairwiseFunction, X::Matrix, Y::Matrix)

    In-place version of ``kernelmatrix`` where pre-allocated matrix ``K`` will 
    be overwritten with the kernel matrix.
    
    
----------------
Hyper Parameters
----------------

The submodule ``HyperParameters`` defines a ``HyperParameter`` type as well as a
``Bound`` and ``Interval`` type. The hyper parameter type stores the current
value of a hyper parameter as well as an interval that defines the domain of
the hyper parameter. Each ``Kernel`` type is a struct of ``HyperParameter``
instances.

Often, hyper parameter values are restricted to an interval with an open bounded
start point or end point (ex. :math:`\alpha > 0`). Exclusive finite endpoints 
such as these are often disallowed in optimization algorithms. This module 
includes two transformations to work around these constraints:

 * ``theta``: The function :math:`\theta` is used to transform a parameter
   restricted to a finite open-bounded interval to an interval without finite
   open bounds.

 * ``eta``: The function :math:`\eta` is the inverse of :math:`\theta`. It 
   converts from values in the transformed space back to the original parameter 
   space.

The specific form of :math:`\theta` and :math:`\eta` depends on the interval
that the parameter is restricted to. Given finite :math:`a`, finite
:math:`b` and parameter :math:`\alpha`, functions :math:`\theta` and 
:math:`\eta` are  defined as follows:

=================================== =============================================== ====================================== ========================================================
Domain :math:`\alpha`               Function :math:`\theta_\alpha = \theta(\alpha)`    Domain :math:`\theta_\alpha`           Function :math:`\eta\left(\theta_{\alpha}\right)`
=================================== =============================================== ====================================== ========================================================
:math:`\left(a,b\right)`            :math:`\log(\alpha-a) - \log(b - \alpha)`       :math:`\left(-\infty,\infty\right)`    :math:`(b\exp(\theta_\alpha)+a)/(1+\exp(\theta_\alpha))`
:math:`\left(a,b\right]`            :math:`\log(\alpha-a)`                          :math:`\left(-\infty,\log(b-a)\right]` :math:`\exp(\theta_\alpha) + a`
:math:`\left[a,b\right)`            :math:`\log(b-\alpha)`                          :math:`\left(-\infty,\log(b-a)\right]` :math:`b - \exp(\theta_\alpha)`
:math:`\left(a,\infty\right)`       :math:`\log(\alpha - a)`                        :math:`\left(-\infty,\infty\right)`    :math:`\exp(\theta_\alpha) + a`
:math:`\left(-\infty,b\right)`      :math:`\log(b - \alpha)`                        :math:`\left(-\infty,\infty\right)`    :math:`b - \exp(\theta_\alpha)`
:math:`\left(-\infty,\infty\right)` N/A                                             N/A                                    N/A
:math:`\left[a,b\right]`            N/A                                             N/A                                    N/A
:math:`\left(-\infty,b\right]`      N/A                                             N/A                                    N/A
:math:`\left[a,\infty\right)`       N/A                                             N/A                                    N/A
=================================== =============================================== ====================================== ========================================================

The following functions are supported by the hyper parameter submodule:

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
    parameter may take on.

    The ``Interval`` type is also used to define the form of ``theta``.

.. function:: interval(a::Union{Bound,Void},b::Union{Bound,Void}) -> Interval

    Constructs an ``Interval`` type. If ``nothing`` is provided for ``a`` or
    ``b``, then a ``NullBound`` will be substituted. If both ``a`` and ``b`` are
    ``nothing``, the interval defaults to an unbounded ``Interval{Float64}``
    type.

.. function:: HyperParameter(a::Real, I::Interval) -> HyperParameter

    Constructs a hyper parameter with value ``a`` and domain restriction ``I``.
    If ``a`` is an invalid value for ``I``, then the constructor will fail.

.. function:: checkvalue(P::HyperParameter, x::Real)

    Checks if ``x`` falls within the hyper parameter domain of ``P``.

.. function:: getvalue(P::HyperParameter)

    Gets the current value of hyper parameter ``P``.

.. function:: setvalue!(P::HyperParameter, x::Real)

    Sets the value of ``P`` to ``x``.

.. function:: checktheta(P::HyperParameter, x::Real)

    Checks if :math:`\eta(x)` falls within the hyper parameter domain of ``P``.

.. function:: gettheta(P::HyperParameter)

    Gets the current value of :math:`\theta(P)` of hyper parameter ``P``.

.. function:: settheta!(P::HyperParameter, x::Real)

    Sets the value of ``P`` to :math:`\eta(x)`.


.. _design matrices: https://en.wikipedia.org/wiki/Design_matrix

.. _MLKernels.jl: https://github.com/trthatcher/MLKernels.jl
