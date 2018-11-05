import Base: @deprecate, depwarn

abstract type MemoryLayout end

struct ColumnMajor <: MemoryLayout
    function ColumnMajor()
        depwarn("Use `Val(:col)` instead of `ColumnMajor()`", :ColumnMajor)
        new()
    end
end

struct RowMajor <: MemoryLayout
    function RowMajor()
        depwarn("Use `Val(:row)` instead of `RowMajor()`", :RowMajor)
        new()
    end
end

@deprecate RowMajor Val{:row}
@deprecate ColumnMajor Val{:col}

layout_map(orient) = typeof(orient) <: RowMajor ? Val(:row) : Val(:col)

col_warn = "Use `Val(:col)` instead of `ColumnMajor()`"
row_warn = "Use `Val(:row)` instead of `RowMajor()`"

function kernelmatrix!(
        σ::MemoryLayout,
        P::Matrix,
        κ::Kernel,
        X::AbstractMatrix,
        symmetrize::Bool = true
    )
    orientation = layout_map(σ)
    kernelmatrix!(orientation, P, κ, X, symmetrize)
end

function kernelmatrix!(
        σ::MemoryLayout,
        P::Matrix,
        κ::Kernel,
        X::AbstractMatrix,
        Y::AbstractMatrix
    )
    orientation = layout_map(σ)
    kernelmatrix!(orientation, P, κ, X, Y)
end

function kernelmatrix(
        σ::MemoryLayout,
        κ::Kernel,
        X::AbstractMatrix,
        symmetrize::Bool = true
    )
    orientation = layout_map(σ)
    kernelmatrix(orientation, κ, X, symmetrize)
end

function kernelmatrix(
        σ::MemoryLayout,
        κ::Kernel,
        X::AbstractMatrix,
        Y::AbstractMatrix
    )
    orientation = layout_map(σ)
    kernelmatrix(orientation, κ, X, Y)
end