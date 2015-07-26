Base Kernels
============

The ``BaseKernel`` type is a subtype of ``StandardKernel``:

.. code-block:: julia

    abstract BaseKernel{T<:FloatingPoint} <: StandardKernel{T}

Base kernels serve as the building  blocks of more complex kernels. Each base kernel is defined by 
a ``phi`` function (this is not exported by the package) which is applied in a specific way
depending on the subtype of the base kernel instance.

The majority of kernels used in practice can be expressed by an element-wise operation that is
summed over the dimensions of the input vector (and then transformed by a scalar function).
Formally:

.. math::
    
    k(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n \phi(x_i,y_i) \qquad \phi:\mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}

In Julia, these kernels are a subtype of the additive kernel abstract type:

.. code-block:: julia

    abstract AdditiveKernel{T<:FloatingPoint} <: BaseKernel{T}

Separable kernels have the further property :math:`\phi(x,y) = h(x)h(y)`. This means that 
:math:`h` is a preprocessing operation applied to each elemented of the input vectors:

.. code-block:: julia

    abstract SeparableKernel{T<:FloatingPoint} <: AdditiveKernel{T}

Additive kernels can be extended using Automatic Relevance Determination (ARD). In this package, ARD is defined formally by:

.. math::

    k(\mathbf{x},\mathbf{y};\mathbf{w}) = \sum_{i=1}^n w_i \phi(x_i,y_i) \qquad \phi:\mathbb{R}^2 \rightarrow \mathbb{R}, \; w_i > 0 \; \forall i


The base kernels defined by default are listed below:

----------------
Additive Kernels
----------------

See page 79 of [berg]_ for examples of other negative or positive definite kernels.

.. function:: SquaredDistanceKernel{T<:FloatingPoint}(t::T = 1.0)

    Construct a squared distance kernel type:

    .. math::
    
        \kappa(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n (x_i - y_i)^{2t} \qquad 0 < t \leq 1

    The squared distance is a **negative definite** kernel [berg]_. The radial basis kernel is a scalar
    transformation of this kernel.

.. function:: SineSquaredKernel{T<:FloatingPoint}(t::T = 1.0)

    Construct a sine squared kernel type:

    .. math::
    
        \kappa(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n \sin^{2t}(x_i - y_i) \qquad 0 < t \leq 1

    The sine squared kernel is a **negative definite** kernel [berg]_. The periodic kernel is a scalar
    transformation of this kernel.

.. function:: ChiSquaredKernel{T<:FloatingPoint}(t::T = 1.0)

    Construct a Chi-Squared kernel:

    .. math::
    
        \kappa(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n \left(\frac{(x_i - y_i)^2}{x_i + y_i}\right)^t \qquad 0 < t \leq 1, \; x_i > 0 \; \forall i, \; y_i > 0 \; \forall i

    The Chi-Squared kernel is often used with bag-of-words models.

-----------------
Separable Kernels
-----------------

Since separable kernels are equivalent to a vector dot product, they are all **mercer** kernels:

.. function:: ScalarProductKernel{T<:FloatingPoint}()

    Construct a Scalar Product kernel:

    .. math::
    
        \kappa(\mathbf{x},\mathbf{y}) = \mathbf{x}^{\intercal} \mathbf{y}

    This is simply the scalar product of two vectors.

.. function:: MercerSigmoidKernel{T<:FloatingPoint}()

    Construct a Mercer sigmoid kernel:

    .. math::
    
        \kappa(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n \tanh\left(\frac{x_i-d}{b}\right) \tanh\left(\frac{y_i-d}{b}\right) \qquad b > 0

---------------------------------
Automatic Relevance Determination
---------------------------------

The ``ismercer`` and ``isnegdef`` functions for ARD evaluate to true if the underlying kernel is Mercer or negative definite, respectively.

.. function:: ARD{T<:FloatingPoint}(κ::AdditiveKernel{T}, w::Vector{T})

    Construct an automatic relevance determination kernel:

    .. math::
    
        \kappa(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n w_i\phi(x_i,y_i) \qquad \phi \text{ is a kernel in } \mathbb{R}, \; w_i > 0 \; \forall i
