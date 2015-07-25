Base Kernels
============

**Base Kernels** are the most simple building blocks for kernels. In Julia, we have:

.. code-block:: julia

    abstract BaseKernel{T<:FloatingPoint} <: StandardKernel{T}

Every base kernel is defined by a ``kappa`` function. The :math:`\kappa` function is applied in a certain way depending on the kernel subtype to define the kernel function :math:`k`.


Additive Kernels
----------------

The abstract type ``AdditiveKernel`` has been inplemented for kernels of the form:

.. math::
    
    k(x,y) = \sum_{i=1}^n \kappa(x_i,y_i)

Where :math:`\kappa:\mathbb{R}^2 \rightarrow \mathbb{R}` is a function applied to pairs of elements of :math:`x` and :math:`y`.

Squared Distance Kernel
^^^^^^^^^^^^^^^^^^^^^^^

.. function:: SquaredDistanceKernel{T<:FloatingPoint}(t::T = 1.0)

    Create a squared distance kernel type. The Squared Distance kernel is defined by:

    .. math::
    
        \kappa(a,b) = (a-b)^2 \quad \implies \quad k(x,y) = \sum_{i=1}^n (x_i - y_i)^{2t}

    where :math:`0 < t \leq 1`.
