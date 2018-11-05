# MLKernels 0.4.0
* Updated to support julia 0.7
* `PairwiseFunction` renamed to `BaseFunction` and no longer exported
* `HyperParameters` deprecated
* `MemoryLayout` types (`ColumnMajor` & `RowMajor`) deprecated and replaced with `Val(:row)` and `Val(:col)`

# MLKernels 0.3.0
* Updated to support julia 0.6

# MLKernels 0.2.0

* Major rewrite of entire package
* Added `Parameter` type for hyper parameters
* Redefined all kernels as collections of `Parameter`s internally
