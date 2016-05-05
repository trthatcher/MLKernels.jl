#===================================================================================================
  Generic pairwise functions for kernels consuming two vectors
===================================================================================================#

pairwise{T}(κ::StandardKernel{T}, x::T, y::T) = phi(κ, x, y)
pairwise{T}(κ::StandardKernel{T}, x::AbstractVector{T}, y::AbstractVector{T}) = phi(κ, x, y)

pairwise{T}(κ::AdditiveKernel{T}, x::T, y::T) = phi(κ, x, y)
function pairwise{T}(κ::AdditiveKernel{T}, x::AbstractVector{T}, y::AbstractVector{T})
    s = zero(T)
    @inbounds for i in eachindex(x)
        s += phi(κ, x[i], y[i])
    end
    s
end

for (scheme, dimension) in ((:(:row), 1), (:(:col), 2))
    @eval begin

        function subvector(::Type{Val{$scheme}}, X::AbstractMatrix,  i::Integer)
            $(scheme == :(:row) ? :(slice(X, i, :)) : :(slice(X, :, i)))
        end

        @inline function init_pairwise{T}(
                 ::Type{Val{$scheme}},
                X::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(X,$dimension))
        end

        @inline function init_pairwise{T}(
                 ::Type{Val{$scheme}},
                X::AbstractMatrix{T}, 
                Y::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(Y,$dimension))
        end

        function pairwise!{T}(
                 ::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::PairwiseKernel{T},
                X::AbstractMatrix{T}
            )
            if !((n = size(X, $dimension)) == size(K,1) == size(K,2))
                errorstring = string("Dimensions of K must match dimension ", $dimension, "of X")
                throw(DimensionMismatch(errorstring))
            end
            for j = 1:n
                xj = subvector(Val{$scheme}, X, j)
                for i = 1:j
                    xi = subvector(Val{$scheme}, X, i)
                    K[i,j] = pairwise(κ, xi, xj)
                end
            end
            LinAlg.copytri!(K, 'U', false)
        end

        function pairwise!{T}(
                 ::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::PairwiseKernel{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T},
            )
            if !((n = size(X, $dimension)) == size(K,1))
                errorstring = string("Dimension 1 of K must match dimension ", $dimension, "of X")
                throw(DimensionMismatch(errorstring))
            elseif !((m = size(Y, $dimension)) == size(K,2))
                errorstring = string("Dimension 1 of K must match dimension ", $dimension, "of Y")
                throw(DimensionMismatch(errorstring))
            end
            for j = 1:m
                yj = subvector(Val{$scheme}, Y, j)
                for i = 1:n
                    xi = subvector(Val{$scheme}, X, i)
                    K[i,j] = pairwise(κ, xi, yj)
                end
            end
            K
        end
    end
end

function pairwise{T}(
        v::DataType,
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T}
    )
    pairwise!(v, init_pairwise(v, X), κ, X)
end

function pairwise{T}(
        v::DataType,
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwise!(v, init_pairwise(v, X, Y), κ, X, Y)
end


#===================================================================================================
  ScalarProduct and SquaredDistance using BLAS
===================================================================================================#

function squared_distance!{T<:AbstractFloat}(G::Matrix{T}, xᵀx::Vector{T})
    if !((n = length(xᵀx)) == size(G,1) == size(G,2))
        throw(DimensionMismatch("Gramian matrix must be square."))
    end
    @inbounds for j = 1:n, i = (1:j)
        G[i,j] = xᵀx[i] - 2G[i,j] + xᵀx[j]
    end
    LinAlg.copytri!(G, 'U')
end

function squared_distance!{T<:AbstractFloat}(G::Matrix{T}, xᵀx::Vector{T}, yᵀy::Vector{T})
    if size(G,1) != length(xᵀx)
        throw(DimensionMismatch(""))
    elseif size(G,2) != length(yᵀy)
        throw(DimensionMismatch(""))
    end
    @inbounds for I in CartesianRange(size(G))
        G[I] = xᵀx[I[1]] - 2G[I] + yᵀy[I[2]]
    end
    G
end

for (scheme, dimension) in ((:(:row), 1), (:(:col), 2))
    @eval begin

        function dotvectors!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}},
                xᵀx::Vector{T},
                X::AbstractMatrix{T}
            )
            if !(size(X,$dimension) == length(xᵀx))
                errorstring = string("Dimension mismatch on dimension ", $dimension)
                throw(DimensionMismatch(errorstring))
            end
            fill!(xᵀx, zero(T))
            for I in CartesianRange(size(X))
                xᵀx[I.I[$dimension]] += X[I]^2
            end
            xᵀx
        end

        @inline function dotvectors{T<:AbstractFloat}(::Type{Val{$scheme}}, X::AbstractMatrix{T})
            dotvectors!(Val{$scheme}, Array(T, size(X,$dimension)), X)
        end

        @inline function gramian!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}},
                G::Matrix{T},
                X::AbstractMatrix{T}
            )
            gramian!(Val{$scheme}, G, X, X)
        end

        @inline function gramian!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}}, 
                G::Matrix{T}, 
                X::AbstractMatrix{T}, 
                Y::AbstractMatrix{T}
            )
            $(scheme == :(:row) ? :A_mul_Bt! : :At_mul_B!)(G, X, Y)
        end

        @inline function pairwise!{T}(
                 ::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::ScalarProductKernel{T},
                X::AbstractMatrix{T}
            )
            gramian!(Val{$scheme}, K, X)
        end

        @inline function pairwise!{T}(
                 ::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::ScalarProductKernel{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T},
            )
            gramian!(Val{$scheme}, K, X, Y)
        end

        @inline function pairwise!{T}(
                 ::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::SquaredDistanceKernel{T},
                X::AbstractMatrix{T}
            )
            gramian!(Val{$scheme}, K, X)
            xᵀx = dotvectors(Val{$scheme}, X)
            squared_distance!(K, xᵀx)
        end

        @inline function pairwise!{T}(
                 ::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::SquaredDistanceKernel{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T},
            )
            gramian!(Val{$scheme}, K, X, Y)
            xᵀx = dotvectors(Val{$scheme}, X)
            yᵀy = dotvectors(Val{$scheme}, Y)
            squared_distance!(K, xᵀx, yᵀy)
        end
    end
end
