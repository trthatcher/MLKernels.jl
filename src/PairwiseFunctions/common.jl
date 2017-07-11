for layout in (RowMajor, ColumnMajor)

    isrowmajor = layout == RowMajor
    dim_obs, dim_param = isrowmajor ? (1, 2) : (2, 1)
    NT, TN = isrowmajor ? ('N', 'T') : ('T', 'N')

    @eval begin

        function dotvectors!(
                 ::$layout,
                xᵀx::Vector{T},
                X::Matrix{T}
            ) where {T<:AbstractFloat}
            if !(size(X,$dim_obs) == length(xᵀx))
                errorstring = string("Dimension mismatch on dimension ", $dim_obs)
                throw(DimensionMismatch(errorstring))
            end
            fill!(xᵀx, zero(T))
            for I in CartesianRange(size(X))
                xᵀx[I.I[$dim_obs]] += X[I]^2
            end
            xᵀx
        end

        @inline function dotvectors(σ::$layout, X::Matrix{T}) where {T<:AbstractFloat}
            dotvectors!(σ, Array{T}(size(X,$dim_obs)), X)
        end

        function gramian!(
                 ::$layout,
                G::Matrix{T},
                X::Matrix{T},
                symmetrize::Bool
            ) where {T<:BLAS.BlasReal}
            BLAS.syrk!('U', $NT, one(T), X, zero(T), G)
            symmetrize ? LinAlg.copytri!(G, 'U') : G
        end

        @inline function gramian!(
                 ::$layout, 
                G::Matrix{T}, 
                X::Matrix{T}, 
                Y::Matrix{T}
            ) where {T<:BLAS.BlasReal}
            BLAS.gemm!($NT, $TN, one(T), X, Y, zero(T), G)
        end
    end
end

function squared_distance!(G::Matrix{T}, xᵀx::Vector{T}, symmetrize::Bool) where {T<:AbstractFloat}
    if !((n = length(xᵀx)) == size(G,1) == size(G,2))
        throw(DimensionMismatch("Gramian matrix must be square."))
    end
    @inbounds for j = 1:n, i = (1:j)
        G[i,j] = xᵀx[i] - 2G[i,j] + xᵀx[j]
    end
    symmetrize ? LinAlg.copytri!(G, 'U') : G
end

function squared_distance!(G::Matrix{T}, xᵀx::Vector{T}, yᵀy::Vector{T}) where {T<:AbstractFloat}
    if size(G,1) != length(xᵀx)
        throw(DimensionMismatch("Length of xᵀx must match rows of G"))
    elseif size(G,2) != length(yᵀy)
        throw(DimensionMismatch("Length of yᵀy must match columns of G"))
    end
    @inbounds for I in CartesianRange(size(G))
        G[I] = xᵀx[I[1]] - 2G[I] + yᵀy[I[2]]
    end
    G
end
