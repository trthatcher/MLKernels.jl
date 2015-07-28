Base Kernels
============

The ``BaseKernel`` type is a subtype of ``StandardKernel``:

.. code-block:: julia

    abstract BaseKernel{T<:FloatingPoint} <: StandardKernel{T}

Base kernels serve as the building  blocks of more complex kernels. Each base kernel is defined by 
a ``phi`` function (this is not exported by the package) which is applied in a specific way
depending on the subtype of the base kernel instance.

.. contents::
    :local:
    :backlinks: none

Additive Kernels
----------------

The majority of kernels used in practice can be expressed by an element-wise operation that is
summed over the dimensions of the input vector (and then transformed by a scalar function).
Formally:

.. math::
    
    k(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n \phi(x_i,y_i) \qquad \phi:\mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}

In Julia, these kernels are a subtype of the additive kernel abstract type:

.. code-block:: julia

    abstract AdditiveKernel{T<:FloatingPoint} <: BaseKernel{T}

Squared Distance Kernel
~~~~~~~~~~~~~~~~~~~~~~~

Construct a squared distance kernel type:

.. math::
    
    \kappa(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n (x_i - y_i)^{2t} \qquad 0 < t \leq 1

The squared distance is a **negative definite** kernel [berg]_. The radial basis kernel is a scalar 
transformation of this kernel.

.. code-block:: julia

    SquaredDistanceKernel()   # Squared distance kernel with t = 1.0
    SquaredDistanceKernel(t)  # Squared distance kernel specified t value


Sine Squared Kernel
~~~~~~~~~~~~~~~~~~~
    
The sine squared kernel is a **negative definite** kernel [berg]_. It can be used to construct the
periodic kernel:

.. math::
    
    \kappa(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n \sin^{2t}(x_i - y_i) \qquad 0 < t \leq 1

.. code-block:: julia

    SineSquaredKernel()   # Sine Squared kernel with t = 1.0
    SineSquaredKernel(t)  # Sine Squared kernel specified t value


Chi-Squared Kernel
~~~~~~~~~~~~~~~~~~

The Chi-Squared kernel is often used with bag-of-words models.

.. math::
    
    \kappa(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n \left(\frac{(x_i - y_i)^2}{x_i + y_i}\right)^t \qquad 0 < t \leq 1, \; x_i > 0 \; \forall i, \; y_i > 0 \; \forall i

.. code-block:: julia

    ChiSquaredKernel()   # Sine Squared kernel with t = 1.0
    ChiSquaredKernel(t)  # Sine Squared kernel specified t value


Separable Kernels
~~~~~~~~~~~~~~~~~

Separable kernels have the further property :math:`\phi(x,y) = h(x)h(y)`. This means that 
:math:`h` is a preprocessing operation applied to each elemented of the input vectors:

.. code-block:: julia

    abstract SeparableKernel{T<:FloatingPoint} <: AdditiveKernel{T}

Additive kernels can be extended using Automatic Relevance Determination (ARD). In this package, ARD is defined formally by:

.. math::

    k(\mathbf{x},\mathbf{y};\mathbf{w}) = \sum_{i=1}^n w_i \phi(x_i,y_i) \qquad \phi:\mathbb{R}^2 \rightarrow \mathbb{R}, \; w_i > 0 \; \forall i


Scalar Product Kernel
^^^^^^^^^^^^^^^^^^^^^

This is simply the scalar product of two vectors.

.. math::
    
    \kappa(\mathbf{x},\mathbf{y}) = \mathbf{x}^{\intercal} \mathbf{y}

.. code-block:: julia

    ScalarProductKernel()            # Default Float64 Scalar Product
    ScalarProductKernel{Float32}()   # Float32 Scalar Product
    ScalarProductKernel{Float64}()   # Float64 Scalar Product
    ScalarProductKernel{BigFloat}()  # BigFloat Scalar Product


Mercer Sigmoid Kernel
^^^^^^^^^^^^^^^^^^^^^

Construct a Mercer sigmoid kernel:

.. math::
    
    \kappa(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n \tanh\left(\frac{x_i-d}{b}\right) \tanh\left(\frac{y_i-d}{b}\right) \qquad b > 0


Automatic Relevance Determination
---------------------------------

Automatic relevance determination extends the additive kernels by adding a weight to each of the 
functions applied to the elements of the input vectors:

.. math::
    
    \kappa(\mathbf{x},\mathbf{y}) = \sum_{i=1}^n w_i\phi(x_i,y_i) \qquad \phi \text{ is a kernel in } \mathbb{R}, \; w_i > 0 \; \forall i

To construct an ARD kernel in Julia:

.. code-block:: julia

    ARD(ScalarProductKernel(), [1.0, 2.0])    # Create a Scalar Product ARD kernel for 2-dimensional vectors
    ARD(ChiSquaredKernel(), [1.0, 2.0, 3.0])  # Create a Chi-Squared ARD kernel for 3-dimensional vectors

The ``ismercer`` and ``isnegdef`` functions for ARD evaluate to true if the underlying kernel is 
Mercer or negative definite, respectively.

