Composite Kernels
=================

The ``CompositeKernel`` type is a subtype of ``StandardKernel``:

.. code-block:: julia

    abstract CompositeKernel{T<:FloatingPoint} <: StandardKernel{T}

Composite kernels extend the base kernels by using a scalar transformation to construct more complex kernels. Formally:

.. math::
    
    k(\mathbf{x},\mathbf{y}) =  \phi(\kappa(x_i,y_i)) \qquad \kappa:\mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}

Blah

.. function:: ExponentialKernel{T<:FloatingPoint}(κ::BaseKernel{T} = SquaredDistanceKernel(1.0), α::T = one(T), γ::T = one(T))

    Construct an exponential class kernel:

    .. math::
    
        k(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha k_\kappa(\mathbf{x},\mathbf{y})^{2 \lambda}\right) \qquad \alpha > 0, \; 0 < \lambda \leq 1

    where :math:`k_\kappa` is a non-negative negative definite kernel. When :math:`k_\kappa` is the squared distance kernel, then :math:`k` is the radial basis (Gaussian) kernel.
