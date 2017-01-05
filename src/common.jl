function promote_type_float(T_i::DataType...)
    T_max = promote_type(T_i...)
    T_max <: AbstractFloat ? T_max : Float64
end

function promote_type_int(U_i::DataType...)
    U_max = promote_type(U_i...)
    U_max <: Signed ? U_max : Int64
end

for layout in (RowMajor, ColumnMajor)

    isrowmajor = layout == RowMajor
    dim_obs    = isrowmajor ? 1 : 2
    dim_param  = isrowmajor ? 2 : 1

    @eval begin

        function dotvectors!{T<:AbstractFloat}(
                 ::$layout,
                xᵀx::Vector{T},
                X::Matrix{T}
            )
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

        @inline function dotvectors{T<:AbstractFloat}(σ::$layout, X::Matrix{T})
            dotvectors!(σ, Array(T, size(X,$dim_obs)), X)
        end

        function gramian!{T<:AbstractFloat}(
                 ::$layout,
                G::Matrix{T},
                X::Matrix{T},
                symmetrize::Bool
            )
            LinAlg.syrk_wrapper!(G, $(isrowmajor ? 'N' : 'T'), X)
            symmetrize ? LinAlg.copytri!(G, 'U') : G
        end

        @inline function gramian!{T<:AbstractFloat}(
                 ::$layout, 
                G::Matrix{T}, 
                X::Matrix{T}, 
                Y::Matrix{T}
            )
            LinAlg.gemm_wrapper!(G, $(isrowmajor ? 'N' : 'T'), $(isrowmajor ? 'T' : 'N'), X, Y)
        end
    end
end

function squared_distance!{T<:AbstractFloat}(G::Matrix{T}, xᵀx::Vector{T}, symmetrize::Bool)
    if !((n = length(xᵀx)) == size(G,1) == size(G,2))
        throw(DimensionMismatch("Gramian matrix must be square."))
    end
    @inbounds for j = 1:n, i = (1:j)
        G[i,j] = xᵀx[i] - 2G[i,j] + xᵀx[j]
    end
    symmetrize ? LinAlg.copytri!(G, 'U') : G
end

function squared_distance!{T<:AbstractFloat}(G::Matrix{T}, xᵀx::Vector{T}, yᵀy::Vector{T})
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
