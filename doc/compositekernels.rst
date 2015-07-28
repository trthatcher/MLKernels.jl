Composite Kernels
=================

The ``CompositeKernel`` type is a subtype of ``StandardKernel``:

.. code-block:: julia

    abstract CompositeKernel{T<:FloatingPoint} <: StandardKernel{T}

Composite kernels extend the base kernels by using a scalar transformation to construct more 
complex kernels. Formally:

.. math::
    
    \kappa(\mathbf{x},\mathbf{y}) =  \phi(\kappa(\mathbf{x},\mathbf{y})) \qquad \kappa:\mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}

.. contents::
    :local:
    :backlinks: none

Exponential Kernel
------------------

.. function:: ExponentialKernel{T<:FloatingPoint}(κ::BaseKernel{T} = SquaredDistanceKernel(1.0), α::T = one(T), γ::T = one(T))

    Construct an exponential class kernel:

    .. math::
    
        k(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha \kappa(\mathbf{x},\mathbf{y})^{\gamma}\right) \qquad \alpha > 0, \; 0 < \gamma \leq 1

    where :math:`\kappa` is a non-negative negative definite kernel. When :math:`\kappa` is the
    squared distance kernel, then :math:`k` is the radial basis (Gaussian) kernel.

Rational-Quadratic Kernel
-------------------------

.. function:: RationalQuadraticKernel{T<:FloatingPoint}(
                κ::BaseKernel{T} = SquaredDistanceKernel(1.0), 
                α::T = one(T), 
                β::T = one(T),
                 γ::T = one(T))

    Construct a rational-quadratic class kernel:

    .. math::
    
        k(\mathbf{x},\mathbf{y}) = \left(1 +\alpha \kappa(\mathbf{x},\mathbf{y})^{\gamma}\right)^{-\beta} \qquad \alpha > 0, \; \beta > 0, \; 0 < \gamma \leq 1

    where :math:`\kappa` is a non-negative negative definite kernel. The rational-quadratic
    kernel is a Mercer kernel.

Matern Kernel
-------------

.. function:: MaternKernel{T<:FloatingPoint}(κ::BaseKernel{T} = SquaredDistanceKernel(1.0), α::T = one(T), β::T = one(T), γ::T = one(T))

    Construct a Matern class kernel:

    .. math::

        k(\mathbf{x},\mathbf{y}) = \frac{1}{2^{\nu-1}\Gamma(\nu)} \left(\frac{2\sqrt{\nu}\kappa(\mathbf{x},\mathbf{y})}{\theta}\right)^{\nu} K_{\nu}\left(\frac{2\sqrt{\nu}\kappa(\mathbf{x},\mathbf{y})}{\theta}\right)
    
    where :math:`\kappa` is a non-negative negative definite kernel, :math:`\Gamma` is the gamma
    function, :math:`K_{\nu}` is the modified Bessel function of the second kind, :math:`\nu > 0`
    and :math:`\theta > 0`. The Matern kernel is a Mercer kernel. 

Power Kernel
------------

.. function:: PowerKernel{T<:FloatingPoint}(κ::BaseKernel{T} = SquaredDistanceKernel(1.0), γ::T = one(T))

    Construct a power kernel:

    .. math::
    
        k(\mathbf{x},\mathbf{y}) = \kappa(\mathbf{x},\mathbf{y})^{\gamma} \qquad 0 < \gamma \leq 1

    where :math:`\kappa` is a non-negative negative definite kernel. The power kernel is a
    negative definite kernel.

.. function:: LogKernel{T<:FloatingPoint}(κ::BaseKernel{T} = SquaredDistanceKernel(1.0), α::T = one(T), γ::T = one(T))

    Construct a log kernel:

    .. math::
    
        k(\mathbf{x},\mathbf{y}) = \log(1 + \alpha\kappa(\mathbf{x},\mathbf{y})^{\gamma}) \qquad \alpha > 0, \; 0 < \gamma \leq 1

    where :math:`\kappa` is a non-negative negative definite kernel. The power kernel is a
    negative definite kernel.

.. function:: PolynomialKernel{T<:FloatingPoint}(κ::BaseKernel{T} = ScalarProductKernel(), α::T = one(T), c::T = one(T), d::T = convert(T,2))

    Construct a polynomial kernel:

    .. math::
    
        k(\mathbf{x},\mathbf{y}) = (\alpha\kappa(\mathbf{x},\mathbf{y}) + c)^d \qquad \alpha > 0, \; c \geq 0, \; d \in \mathbb{Z}_{+}

    where :math:`\kappa` is a Mercer kernel. The polynomial kernel is a Mercer kernel.

.. function:: ExponentiatedKernel{T<:FloatingPoint}(κ::BaseKernel{T} = ScalarProductKernel(), α::T = one(T))

    Construct an exponentiated kernel:

    .. math::
    
        k(\mathbf{x},\mathbf{y}) = \exp(\alpha\kappa(\mathbf{x},\mathbf{y}) + c) \qquad \alpha > 0, \; c \geq 0

    where :math:`\kappa` is a Mercer kernel. An exponentiated kernel is a Mercer kernel.

.. function:: SigmoidKernel{T<:FloatingPoint}(κ::BaseKernel{T} = ScalarProductKernel(), α::T = one(T), c::T = one(T))

    Construct a sigmoid kernel:

    .. math::
    
        k(\mathbf{x},\mathbf{y}) = \tanh(\alpha\kappa(\mathbf{x},\mathbf{y}) + c) \qquad \alpha > 0, \; c \geq 0

    where :math:`\kappa` is a Mercer kernel. The sigmoid kernel is a not a true kernel, although
    it has been used in application.
