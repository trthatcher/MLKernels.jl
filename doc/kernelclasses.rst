.. _kernelclasses:

--------------------------
Kernel Composition Classes
--------------------------

.. function:: KernelComposition(ϕ,κ)

  The ``KernelComposition`` type is used to construct new kernels. The composite
  type consists of two objects: ``ϕ``, a composition class, and ``κ``, an 
  existing kernel. Mathematically, it constructs a new kernel such that if
  :math:`\phi` is the function composing the kernel :math:`\kappa`, then:

  .. math::

    \Psi(\mathbf{x}, \mathbf{y}) = \phi(\kappa(\mathbf{y}, \mathbf{x}))

The implemented ``CompositionClass`` types are listed below:

.. function:: ExponentialClass(α,γ)

  .. math::

    \Psi(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha \kappa(\mathbf{x},\mathbf{y})^{\gamma}\right) \qquad \alpha > 0, \; 0 < \gamma \leq 1


.. function:: RationalQuadraticClass(α,β,γ)
  
  .. math::

    \Psi(\mathbf{x},\mathbf{y}) = \left(1 +\alpha \kappa(\mathbf{x},\mathbf{y})^{\gamma}\right)^{-\beta} \qquad \alpha > 0, \; \beta > 0, \; 0 < \gamma \leq 1

.. function:: MaternClass(ν,θ)
  
  .. math::

    \Psi(\mathbf{x},\mathbf{y}) = \frac{1}{2^{\nu-1}\Gamma(\nu)} \left(\frac{2\sqrt{\nu}\kappa(\mathbf{x},\mathbf{y})}{\theta}\right)^{\nu} K_{\nu}\left(\frac{2\sqrt{\nu}\kappa(\mathbf{x},\mathbf{y})}{\theta}\right)

.. _polynomialclass:

.. function:: PolynomialClass(a,c,d)

  .. math::

    \Psi(\mathbf{x},\mathbf{y}) = (\alpha\kappa(\mathbf{x},\mathbf{y}) + c)^d \qquad \alpha > 0, \; c \geq 0, \; d \in \mathbb{Z}_{+}

.. _exponentiatedclass:

.. function:: ExponentiatedClass(a,c)

  .. math::

    \Psi(\mathbf{x},\mathbf{y}) = \exp(\alpha\kappa(\mathbf{x},\mathbf{y}) + c) \qquad \alpha > 0, \; c \geq 0

.. _powerclass:

.. function:: PowerClass(a,c,γ)

  .. math::

    \Psi(\mathbf{x},\mathbf{y}) = \kappa(\mathbf{x},\mathbf{y})^{\gamma} \qquad 0 < \gamma \leq 1

.. function:: LogClass(α,γ)

.. math::

    \Psi(\mathbf{x},\mathbf{y}) = \log(1 + \alpha\kappa(\mathbf{x},\mathbf{y})^{\gamma}) \qquad \alpha > 0, \; 0 < \gamma \leq 1

.. _sigmoidclass:

.. function:: SigmoidClass(a,c)

  .. math::

    \Psi(\mathbf{x},\mathbf{y}) = \tanh(\alpha\kappa(\mathbf{x},\mathbf{y}) + c) \qquad \alpha > 0, \; c \geq 0

-----------------
Kernel Operations
-----------------

.. _kernelaffinity:

.. function:: KernelAffinity(a,c,κ)

  The kernel affinity object is an affine transformation of a kernel (both
  Mercer and negative definite):

  .. math::

    \Psi(\mathbf{x}, \mathbf{y}) = a \cdot \kappa(\mathbf{x}, \mathbf{y}) + c

  Given a kernel ``κ``, a ``KernelAffinity`` may be constructed by translating
  or scaling by a positive real number:

  .. code-block:: julia
  
    2 * κ + 1  # Constructs a KernelAffinity(2, 1, κ) object


.. _kernelsum:

.. function:: KernelSum(c,κ...)

  The kernel sum corresponds to the following form:

  .. math::

    \Psi(\mathbf{x}, \mathbf{y}) = c + \sum_{i=1}^n \kappa_i(\mathbf{x},\mathbf{y})

 Both Mercer and negative definite kernels are closed under addition. However,
 Mercer kernels may not be mixed with negative definite kernels.

.. _kernelproduct:

.. function:: KernelProduct(a,κ...)

  The kernel product corresponds to the following form:

  .. math::

    \Psi(\mathbf{x}, \mathbf{y}) = a \prod_{i=1}^n \kappa_i(\mathbf{x},\mathbf{y})

  Only Mercer kernels are closed under multiplication.

