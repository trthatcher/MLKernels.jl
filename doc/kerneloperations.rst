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
