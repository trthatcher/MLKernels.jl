#===================================================================================================
  Kernel Kernels Module
===================================================================================================#

module PairwiseFunctions

export

    # Pairwise Functions
    PairwiseFunction,
        InnerProduct,
            ScalarProduct,
        PreMetric,
            ChiSquared,
            SineSquared,
            Metric,
                SquaredEuclidean,

    # Pairwise Function Properties
    isstationary,
    isisotropic,

    # Orientation
    Orientation,

    MemoryLayout,
    RowMajor,
    ColumnMajor,

    # Pairwise function evaluation
    pairwise,
    pairwisematrix,
    pairwisematrix!

import LinearAlgebra

@doc raw"""
    Orientation

Union of the two `Val` types representing the data matrix orientations:

  1. `Val{:row}` identifies when observation vector corresponds to a row of the data matrix
  2. `Val{:col}` identifies when each observation vector corresponds to a column of the data
     matrix
"""
const Orientation = Union{Val{:row}, Val{:col}}

abstract type MemoryLayout end

@doc raw"""
    ColumnMajor()

Identifies when each observation vector corresponds to a column of the data matrix:

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
"""
struct ColumnMajor <: MemoryLayout end

@doc raw"""
    RowMajor()

Identifies when each observation vector corresponds to a row of the data matrix:

This is commonly used in the field of statistics in the context of [design matrices](https://en.wikipedia.org/wiki/Design_matrix). 
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
"""
struct RowMajor <: MemoryLayout end

include("common.jl")
include("pairwise.jl")
include("pairwisematrix.jl")

end # PairwiseFunctions
