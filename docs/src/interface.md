# Interface

## Storage

[**MLKernels.jl**](https://github.com/trthatcher/MLKernels.jl) allows for data matrices to 
be stored in one of two ways with respect to the observations based on parameters provided 
by the user. In order to specify the ordering used, a subtype of the `MemoryLayout` abstract
type can be provided as a parameter to any methods taking matrices as a parameter:

```@docs
RowMajor
```

```@docs
ColumnMajor
```