# Interface

## Data Orientation
Data matrices may be oriented in one of two ways with respect to the observations.
Functions producing a kernel matrix require an `orient` argument to specify the orientation
of the observations within the provided data matrix.

### Row Orientation (Default)

An orientation of `Val(:row)` identifies when observation vector corresponds to a row of
the data matrix. This is commonly used in the field of statistics in the context of
[design matrices](https://en.wikipedia.org/wiki/Design_matrix).

For example, for data matrix $\mathbf{X}$ consisting of observations $\mathbf{x}_1$,
$\mathbf{x}_2$, $\ldots$, $\mathbf{x}_n$:

```math
\mathbf{X}_{row} =
\begin{bmatrix}
    \leftarrow \mathbf{x}_1 \rightarrow \\
    \leftarrow \mathbf{x}_2 \rightarrow \\
    \vdots \\
    \leftarrow \mathbf{x}_n \rightarrow
\end{bmatrix}
```

When row-major ordering is used, then the kernel matrix of $\mathbf{X}$ will match the
dimensions of $\mathbf{X}^{\intercal}\mathbf{X}$. Similarly, the kernel matrix will match
the dimension of $\mathbf{X}^{\intercal}\mathbf{Y}$ for row-major ordering of data
matrix $\mathbf{X}$ and $\mathbf{Y}$.

### Column Orientation

An orientation of `Val(:col)` identifies when each observation vector corresponds to a
column of the data matrix:

```math
\mathbf{X}_{col} =
\mathbf{X}_{row}^{\intercal} =
\begin{bmatrix}
    \uparrow & \uparrow & & \uparrow  \\
    \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x_n} \\
    \downarrow & \downarrow & & \downarrow
\end{bmatrix}
```

With column-major ordering, the kernel matrix will match the dimensions of
$\mathbf{XX}^{\intercal}$. Similarly, the kernel matrix of data matrices $\mathbf{X}$ and
$\mathbf{Y}$ match the dimensions of $\mathbf{XY}^{\intercal}$.


## Essentials

```@docs
ismercer(::Kernel)
isnegdef(::Kernel)
isstationary(::Kernel)
isisotropic(::Kernel)
kernel(::Kernel{T}, ::Real, ::Real) where T
Orientation
kernelmatrix(::Orientation, ::Kernel{T}, ::AbstractMatrix{T1}, symmetrize::Bool) where {T,T1}
kernelmatrix!(::Orientation, ::Matrix{T}, ::Kernel{T}, ::AbstractMatrix{T}, symmetrize::Bool) where {T<:AbstractFloat}
kernelmatrix(::Orientation, ::Kernel{T}, ::AbstractMatrix{T1}, ::AbstractMatrix{T2}) where {T,T1,T2}
kernelmatrix!(::Orientation, ::Matrix{T}, ::Kernel{T}, ::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T<:AbstractFloat}
centerkernelmatrix!(::Matrix{T}) where {T<:AbstractFloat}
```

## Approximation

In many cases, fast, approximate results is more important than a perfect
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
nystrom
kernelmatrix(::NystromFact{T}) where {T<:LinearAlgebra.BlasReal}
```