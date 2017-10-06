#===================================================================================================
  Generic pairwisematrix functions for kernels consuming two vectors
===================================================================================================#

#================================================
  Generic Pairwise Vector Operation
================================================#

function pairwise{T<:AbstractFloat}(f::PairwiseFunction, x::T, y::T)
    pairwise_return(f, pairwise_aggregate(f, pairwise_initiate(f,T), x, y))
end

# No checks, assumes length(x) == length(y) >= 1
function unsafe_pairwise{T<:AbstractFloat}(
        f::PairwiseFunction,
        x::AbstractArray{T}, 
        y::AbstractArray{T}
    )
    s = pairwise_initiate(f, T)
    @simd for I in eachindex(x,y)
        @inbounds xi = x[I]
        @inbounds yi = y[I]
        s = pairwise_aggregate(f, s, xi, yi)
    end
    pairwise_return(f, s)
end

function pairwise{T<:AbstractFloat}(
        f::PairwiseFunction,
        x::AbstractArray{T},
        y::AbstractArray{T}
    )
    if (n = length(x)) != length(y)
        throw(DimensionMismatch("Arrays x and y must have the same length."))
    elseif n == 0
        throw(DimensionMismatch("Arrays x and y must be at least of length 1."))
    end
    unsafe_pairwise(f, x, y)
end



#================================================
  Generic Pairwise Matrix Calculation
================================================#

for layout in (RowMajor, ColumnMajor)

    isrowmajor = layout == RowMajor
    dim_obs    = isrowmajor ? 1 : 2
    dim_param  = isrowmajor ? 2 : 1

    @eval begin

        @inline function subvector(::$layout, X::AbstractMatrix,  i::Integer)
            $(isrowmajor ? :(view(X, i, :)) : :(view(X, :, i)))
        end

        @inline function allocate_pairwisematrix{T<:AbstractFloat}(
                 ::$layout,
                X::AbstractMatrix{T}
            )
            Array{T}(size(X,$dim_obs), size(X,$dim_obs))
        end

        @inline function allocate_pairwisematrix{T<:AbstractFloat}(
                 ::$layout,
                X::AbstractMatrix{T}, 
                Y::AbstractMatrix{T}
            )
            Array{T}(size(X,$dim_obs), size(Y,$dim_obs))
        end

        function checkdimensions(
                 ::$layout,
                P::Matrix, 
                X::AbstractMatrix
            )
            n = size(P,1)
            if size(P,2) != n
                throw(DimensionMismatch("Pairwise matrix P must be square"))
            elseif size(X, $dim_obs) != n
                errorstring = string("Dimensions of P must match dimension ", $dim_obs, " of X")
                throw(DimensionMismatch(errorstring))
            end
            return n
        end

        function checkdimensions(
                 ::$layout,
                P::Matrix,
                X::AbstractMatrix, 
                Y::AbstractMatrix
            )
            n = size(X, $dim_obs)
            m = size(Y, $dim_obs)
            if n != size(P,1)
                errorstring = string("Dimension 1 of P must match dimension ", $dim_obs, "of X")
                throw(DimensionMismatch(errorstring))
            elseif m != size(P,2)
                errorstring = string("Dimension 2 of P must match dimension ", $dim_obs, "of Y")
                throw(DimensionMismatch(errorstring))
            end
            if size(X, $dim_param) != size(Y, $dim_param)
                errorstring = string("Dimension $($dim_param) of X and Y must match")
                throw(DimensionMismatch(errorstring))
            end
            return (n, m)
        end
    end
end

function pairwisematrix!{T<:AbstractFloat}(
        σ::MemoryLayout,
        P::Matrix{T}, 
        f::PairwiseFunction,
        X::AbstractMatrix{T},
        symmetrize::Bool
    )
    n = checkdimensions(σ, P, X)
    for j = 1:n
        xj = subvector(σ, X, j)
        for i = 1:j
            xi = subvector(σ, X, i)
            @inbounds P[i,j] = unsafe_pairwise(f, xi, xj)
        end
    end
    symmetrize ? LinAlg.copytri!(P, 'U', false) : P
end

function pairwisematrix!{T<:AbstractFloat}(
        σ::MemoryLayout,
        P::Matrix{T}, 
        f::PairwiseFunction,
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T},
    )
    n, m = checkdimensions(σ, P, X, Y)
    for j = 1:m
        yj = subvector(σ, Y, j)
        for i = 1:n
            xi = subvector(σ, X, i)
            @inbounds P[i,j] = unsafe_pairwise(f, xi, yj)
        end
    end
    P
end


function pairwisematrix{T<:AbstractFloat}(
        σ::MemoryLayout,
        f::PairwiseFunction, 
        X::AbstractMatrix{T},
        symmetrize::Bool = true
    )
    pairwisematrix!(σ, allocate_pairwisematrix(σ, X), f, X, symmetrize)
end

function pairwisematrix(
        f::PairwiseFunction,
        X::AbstractMatrix,
        symmetrize::Bool = true
    )
    pairwisematrix(RowMajor(), f, X, symmetrize)
end

function pairwisematrix{T<:AbstractFloat}(
        σ::MemoryLayout,
        f::PairwiseFunction, 
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(σ, allocate_pairwisematrix(σ, X, Y), f, X, Y)
end

function pairwisematrix(
        f::PairwiseFunction,
        X::AbstractMatrix,
        Y::AbstractMatrix
    )
    pairwisematrix(RowMajor(), f, X, Y)
end



#===================================================================================================
  ScalarProduct using BLAS/Built-In methods
===================================================================================================#

@inline function pairwisematrix!{T<:BLAS.BlasReal}(
        σ::MemoryLayout,
        P::Matrix{T}, 
        f::ScalarProduct,
        X::Matrix{T},
        symmetrize::Bool
    )
    gramian!(σ, P, X, symmetrize)
end

@inline function pairwisematrix!{T<:BLAS.BlasReal}(
        σ::MemoryLayout,
        P::Matrix{T}, 
        f::ScalarProduct,
        X::Matrix{T},
        Y::Matrix{T},
    )
    gramian!(σ, P, X, Y)
end



#===================================================================================================
  SquaredDistance using BLAS/Built-In methods
===================================================================================================#

function squared_distance!(G::Matrix{T}, xᵀx::Vector{T}, symmetrize::Bool) where {T<:AbstractFloat}
    if !((n = size(G,1)) == size(G,2))
        throw(DimensionMismatch("Gramian matrix must be square."))
    end
    if length(xᵀx) != n
        throw(DimensionMismatch("Length of xᵀx must match order of G"))
    end
    @inbounds for j = 1:n
        xᵀx_j = xᵀx[j]
        for i = 1:(j-1)
            G[i,j] = (xᵀx[i] + xᵀx_j) - 2G[i,j]
        end
        G[j,j] = zero(T)
    end
    symmetrize ? LinAlg.copytri!(G, 'U') : G
end

function squared_distance!(G::Matrix{T}, xᵀx::Vector{T}, yᵀy::Vector{T}) where {T<:AbstractFloat}
    n, m = size(G)
    if length(xᵀx) != n
        throw(DimensionMismatch("Length of xᵀx must match rows of G"))
    elseif length(yᵀy) != m
        throw(DimensionMismatch("Length of yᵀy must match columns of G"))
    end
    @inbounds for j = 1:m
        yᵀy_j = yᵀy[j]
        for i = 1:n
            G[i,j] = (xᵀx[i] + yᵀy_j) - 2G[i,j]
        end
    end
    G
end

function fix_negatives!(σ::MemoryLayout, D::Matrix{T}, X::Matrix{T}, symmetrize::Bool, ϵ::T=zero(T)) where {T<:AbstractFloat}
    if !((n = size(D,1)) == size(D,2))
        throw(DimensionMismatch("Distance matrix must be square."))
    end
    for j = 1:n
        xj = subvector(σ, X, j)
        for i = 1:(j-1)
            if D[i,j] < ϵ
                xi = subvector(σ, X, i)
                D[i,j] = unsafe_pairwise(SquaredEuclidean(), xi, xj)
            end
        end
    end
    symmetrize ? LinAlg.copytri!(D, 'U') : D
end

function fix_negatives!(σ::MemoryLayout, D::Matrix{T}, X::Matrix{T}, Y::Matrix{T}, ϵ::T=zero(T)) where {T<:AbstractFloat}
    n, m = size(D)
    for j = 1:m
        yj = subvector(σ, Y, j)
        for i = 1:n
            if D[i,j] < ϵ
                xi = subvector(σ, X, i)
                D[i,j] = unsafe_pairwise(SquaredEuclidean(), xi, yj)
            end
        end
    end
    D
end

function pairwisematrix!{T<:BLAS.BlasReal}(
        σ::MemoryLayout,
        P::Matrix{T}, 
        f::SquaredEuclidean,
        X::Matrix{T},
        symmetrize::Bool
    )
    gramian!(σ, P, X, false)
    xᵀx = dotvectors(σ, X)
    squared_distance!(P, xᵀx, false)
    fix_negatives!(σ, P, X, true)
    symmetrize ? LinAlg.copytri!(P, 'U') : P 
end

function pairwisematrix!{T<:BLAS.BlasReal}(
        σ::MemoryLayout,
        P::Matrix{T}, 
        f::SquaredEuclidean,
        X::Matrix{T},
        Y::Matrix{T},
    )
    gramian!(σ, P, X, Y)
    xᵀx = dotvectors(σ, X)
    yᵀy = dotvectors(σ, Y)
    squared_distance!(P, xᵀx, yᵀy)
    fix_negatives!(σ, P, X, Y)
end
