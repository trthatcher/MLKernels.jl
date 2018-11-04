for orientation in (:row, :col)

    row_oriented = orientation == :row
    dim_obs, dim_param = row_oriented ? (1, 2) : (2, 1)
    NT, TN = row_oriented ? ('N', 'T') : ('T', 'N')

    @eval begin

        function dotvectors!(
                ::Val{$(Meta.quot(orientation))},
                xᵀx::Vector{T},
                X::Matrix{T}
            ) where {T<:AbstractFloat}
            if !(size(X, $dim_obs) == length(xᵀx))
                errorstring = string("Dimension mismatch on dimension ", $dim_obs)
                throw(DimensionMismatch(errorstring))
            end
            fill!(xᵀx, zero(T))
            for I in Base.Cartesian.CartesianIndices(size(X))
                xᵀx[I.I[$dim_obs]] += X[I]^2
            end
            xᵀx
        end

        @inline function dotvectors(σ::Val{$(Meta.quot(orientation))}, X::Matrix{T}) where {T<:AbstractFloat}
            dotvectors!(σ, Array{T}(undef, size(X,$dim_obs)), X)
        end

        function gramian!(
                ::Val{$(Meta.quot(orientation))},
                G::Matrix{T},
                X::Matrix{T},
                symmetrize::Bool
            ) where {T<:LinearAlgebra.BLAS.BlasReal}
            LinearAlgebra.BLAS.syrk!('U', $NT, one(T), X, zero(T), G)
            symmetrize ? LinearAlgebra.copytri!(G, 'U') : G
        end

        @inline function gramian!(
                ::Val{$(Meta.quot(orientation))},
                G::Matrix{T},
                X::Matrix{T},
                Y::Matrix{T}
            ) where {T<:LinearAlgebra.BLAS.BlasReal}
            LinearAlgebra.BLAS.gemm!($NT, $TN, one(T), X, Y, zero(T), G)
        end
    end
end
