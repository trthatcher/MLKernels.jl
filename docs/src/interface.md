# MLKernels Interface

## Storage

[**MLKernels.jl**](https://github.com/trthatcher/MLKernels.jl) allows for data matrices to 
be stored in one of two ways with respect to the observations based on parameters provided 
by the user. In order to specify the ordering used, a subtype of the `MemoryLayout` abstract
type can be provided as a parameter to any methods taking matrices as a parameter:

```@docs
RowMajor
ColumnMajor
```

## Essentials

```@docs
ismercer(::Kernel)
isnegdef(::Kernel)
isstationary(::PairwiseFunction)
isisotropic(::PairwiseFunction)
kernel(κ::Kernel{T}, x::Real, y::Real) where T
kernelmatrix(σ::MemoryLayout, κ::Kernel{T}, X::AbstractMatrix{T1}, symmetrize::Bool = true) where {T,T1}
kernelmatrix!(σ::MemoryLayout, P::Matrix{T}, κ::Kernel{T}, X::AbstractMatrix{T}, symmetrize::Bool) where {T<:AbstractFloat}
kernelmatrix!(σ::MemoryLayout, P::Matrix{T}, κ::Kernel{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T<:AbstractFloat}
kernelmatrix(σ::MemoryLayout, κ::Kernel{T}, X::AbstractMatrix{T1}, Y::AbstractMatrix{T2}) where {T,T1,T2}
centerkernelmatrix!(K::Matrix{T}) where {T<:AbstractFloat}
```

## Approximation

In many cases, a fast, approximate results is more important than a perfect
result. The Nystrom method can be used to generate a factorization that can be
used to approximate a large, symmetric kernel matrix. Given data matrix
``\mathbf{X} \in \mathbb{R}^{n \times p}`` (one observation per row) and 
kernel matrix ``\mathbf{K} \in \mathbb{R}^{n \times n}``, the Nystrom method
takes a sample ``S`` of the observations of ``\mathbf{X}`` of size 
``s < n`` and generates a factorization such that:

```math
\mathbf{K} \approx \mathbf{C}^{\intercal}\mathbf{WC}
```

Where ``\mathbf{W}`` is the ``s \times s`` pseudo-inverse of the sample 
kernel matrix based on ``S`` and ``\mathbf{C}`` is a ``s \times n``
matrix.

The Nystrom method uses an eigendecomposition of the sample kernel matrix of 
``\mathbf{X}`` to estimate ``\mathbf{K}``. Generally, the order of 
``\mathbf{K}`` must be quite large and the sampling ratio small (ex. 15% or 
less) for the cost of the computing the full kernel matrix to exceed that of the
eigendecomposition. This method will be more effective for kernels that are not 
a direct function of the dot product as they are not able to make use of BLAS in
computing the full matrix ``\mathbf{K}`` and the cross-over point will occur
for smaller ``\mathbf{K}``.

**MLKernels.jl** implements the Nystrom approximation:

```@docs
NystromFact
```