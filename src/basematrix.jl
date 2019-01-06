# Pairwise Scalar & Vector Operation  ======================================================

@inline base_initiate(::BaseFunction, ::Type{T}) where {T} = zero(T)
@inline base_return(::BaseFunction, s::T) where {T} = s

function base_evaluate(f::BaseFunction, x::T, y::T) where {T<:AbstractFloat}
    base_return(f, base_aggregate(f, base_initiate(f,T), x, y))
end

# Note: no checks, assumes length(x) == length(y) >= 1
function unsafe_base_evaluate(
        f::BaseFunction,
        x::AbstractArray{T},
        y::AbstractArray{T}
    ) where {T<:AbstractFloat}
    s = base_initiate(f, T)
    @simd for I in eachindex(x, y)
        @inbounds xi = x[I]
        @inbounds yi = y[I]
        s = base_aggregate(f, s, xi, yi)
    end
    base_return(f, s)
end

function base_evaluate(
        f::BaseFunction,
        x::AbstractArray{T},
        y::AbstractArray{T}
    ) where {T<:AbstractFloat}
    if (n = length(x)) != length(y)
        throw(DimensionMismatch("Arrays x and y must have the same length."))
    elseif n == 0
        throw(DimensionMismatch("Arrays x and y must be at least of length 1."))
    end
    unsafe_base_evaluate(f, x, y)
end


# Pairwise Matrix Calculation  =============================================================

for orientation in (:row, :col)

    row_oriented = orientation == :row
    dim_obs      = row_oriented ? 1 : 2
    dim_param    = row_oriented ? 2 : 1

    @eval begin

        @inline function subvector(
                ::Val{$(Meta.quot(orientation))},
                X::AbstractMatrix,
                i::Integer
            )
            $(row_oriented ? :(view(X, i, :)) : :(view(X, :, i)))
        end

        @inline function allocate_basematrix(
                ::Val{$(Meta.quot(orientation))},
                X::AbstractMatrix{T}
            ) where {T<:AbstractFloat}
            Array{T}(undef, size(X,$dim_obs), size(X,$dim_obs))
        end

        @inline function allocate_basematrix(
                ::Val{$(Meta.quot(orientation))},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T}
            ) where {T<:AbstractFloat}
            Array{T}(undef, size(X,$dim_obs), size(Y,$dim_obs))
        end

        function checkdimensions(
                ::Val{$(Meta.quot(orientation))},
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
                ::Val{$(Meta.quot(orientation))},
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

function basematrix!(
        σ::Orientation,
        P::Matrix{T},
        f::BaseFunction,
        X::AbstractMatrix{T},
        symmetrize::Bool
    ) where {T<:AbstractFloat}
    n = checkdimensions(σ, P, X)
    for j = 1:n
        xj = subvector(σ, X, j)
        for i = 1:j
            xi = subvector(σ, X, i)
            @inbounds P[i,j] = unsafe_base_evaluate(f, xi, xj)
        end
    end
    symmetrize ? LinearAlgebra.copytri!(P, 'U', false) : P
end

function basematrix!(
        σ::Orientation,
        P::Matrix{T},
        f::BaseFunction,
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T},
    ) where {T<:AbstractFloat}
    n, m = checkdimensions(σ, P, X, Y)
    for j = 1:m
        yj = subvector(σ, Y, j)
        for i = 1:n
            xi = subvector(σ, X, i)
            @inbounds P[i,j] = unsafe_base_evaluate(f, xi, yj)
        end
    end
    P
end


# ScalarProduct using BLAS/Built-In methods ================================================

@inline function basematrix!(
        σ::Orientation,
        P::Matrix{T},
        f::ScalarProduct,
        X::Matrix{T},
        symmetrize::Bool
    ) where {T<:LinearAlgebra.BLAS.BlasReal}
    gramian!(σ, P, X, symmetrize)
end

@inline function basematrix!(
        σ::Orientation,
        P::Matrix{T},
        f::ScalarProduct,
        X::Matrix{T},
        Y::Matrix{T},
    ) where {T<:LinearAlgebra.BLAS.BlasReal}
    gramian!(σ, P, X, Y)
end


# SquaredDistance using BLAS/Built-In methods ==============================================

function squared_distance!(
        G::Matrix{T},
        xᵀx::Vector{T},
        symmetrize::Bool
    ) where {T<:AbstractFloat}
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
    symmetrize ? LinearAlgebra.copytri!(G, 'U') : G
end

function squared_distance!(
        G::Matrix{T},
        xᵀx::Vector{T},
        yᵀy::Vector{T}
    ) where {T<:AbstractFloat}
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

function fix_negatives!(
        σ::Orientation,
        D::Matrix{T},
        X::Matrix{T},
        symmetrize::Bool,
        ϵ::T=zero(T)
    ) where {T<:AbstractFloat}
    if !((n = size(D,1)) == size(D,2))
        throw(DimensionMismatch("Distance matrix must be square."))
    end
    for j = 1:n
        xj = subvector(σ, X, j)
        for i = 1:(j-1)
            if D[i,j] < ϵ
                xi = subvector(σ, X, i)
                D[i,j] = unsafe_base_evaluate(SquaredEuclidean(), xi, xj)
            end
        end
    end
    symmetrize ? LinearAlgebra.copytri!(D, 'U') : D
end

function fix_negatives!(
        σ::Orientation,
        D::Matrix{T},
        X::Matrix{T},
        Y::Matrix{T},
        ϵ::T=zero(T)
    ) where {T<:AbstractFloat}
    n, m = size(D)
    for j = 1:m
        yj = subvector(σ, Y, j)
        for i = 1:n
            if D[i,j] < ϵ
                xi = subvector(σ, X, i)
                D[i,j] = unsafe_base_evaluate(SquaredEuclidean(), xi, yj)
            end
        end
    end
    D
end

function basematrix!(
        σ::Orientation,
        P::Matrix{T},
        f::SquaredEuclidean,
        X::Matrix{T},
        symmetrize::Bool
    ) where {T<:LinearAlgebra.BLAS.BlasReal}
    gramian!(σ, P, X, false)
    xᵀx = dotvectors(σ, X)
    squared_distance!(P, xᵀx, false)
    fix_negatives!(σ, P, X, true)
    symmetrize ? LinearAlgebra.copytri!(P, 'U') : P
end

function basematrix!(
        σ::Orientation,
        P::Matrix{T},
        f::SquaredEuclidean,
        X::Matrix{T},
        Y::Matrix{T},
    ) where {T<:LinearAlgebra.BLAS.BlasReal}
    gramian!(σ, P, X, Y)
    xᵀx = dotvectors(σ, X)
    yᵀy = dotvectors(σ, Y)
    squared_distance!(P, xᵀx, yᵀy)
    fix_negatives!(σ, P, X, Y)
end