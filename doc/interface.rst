Standard Interface
==================




Kernel Functions
----------------

.. function:: kernelmatrix(κ::Kernel{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool, symmetrize::Bool)


Kernel Approximation
--------------------


Kernel Algebra
--------------

.. code-block:: julia

    SineSquaredKernel()   # Sine Squared kernel with t = 1.0
    SineSquaredKernel(t)  # Sine Squared kernel specified t value

