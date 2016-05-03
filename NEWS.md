# MLKernels 0.2.0

* Added `Parameter` type for hyper parameters
* Redefined all kernels as collections of `Parameter`s internally
* Removed ARD type
* Removed kernel approximation

# MLKernels 0.1.1 Release Notes

* Added `CompositionClass` type to represent parametric forms of kernels
* Removed `CompositeKernel` type and added a `KernelComposition` type
    * Composition is now accomplished by calling `KernelComposition` with a `CompositionClass` and a `Kernel` object
* Removed `isnegative` function for inspection

# MLKernels 0.1.0 Release Notes

* Requires Julia 0.4
