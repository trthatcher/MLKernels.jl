Standard Interface
==================




Kernel Functions
----------------

.. function:: kernelmatrix{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, is_trans::Bool = false, store_upper::Bool = true, symmetrize::Bool = true)


Kernel Approximation
--------------------


Kernel Algebra
--------------

.. code-block:: julia

    SineSquaredKernel()   # Sine Squared kernel with t = 1.0
    SineSquaredKernel(t)  # Sine Squared kernel specified t value

