#===================================================================================================
  Generic pairwise functions for kernels consuming two vectors
===================================================================================================#

# NO CHECKS
@inline function vectorpairwise{T<:AbstractFloat}(
        κ::StandardKernel{T},
        x::AbstractArray{T}, 
        y::AbstractArray{T}
    )
    phi(κ, x, y)
end

# NO CHECKS
function vectorpairwise{T<:AbstractFloat}(
        κ::AdditiveKernel{T},
        x::AbstractArray{T}, 
        y::AbstractArray{T}
    )
    s = zero(T)
    @inbounds for i in eachindex(x)
        s += phi(κ, x[i], y[i])
    end
    s
end

for (scheme, dimension) in ((:row, 1), (:col, 2))
    @eval begin

        @inline function subvector(::Type{Val{$scheme}}, X::AbstractMatrix,  i::Integer)
            $(scheme == :row ? :(sub(X, i, :)) : :(sub(X, :, i)))
        end

        @inline function init_pairwise{T<:AbstractFloat}(
                 ::Type{Val{$scheme}},
                X::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(X,$dimension))
        end

        @inline function init_pairwise{T<:AbstractFloat}(
                 ::Type{Val{$scheme}},
                X::AbstractMatrix{T}, 
                Y::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(Y,$dimension))
        end

        function pairwise!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}}
                K::Matrix{T}, 
                κ::PairwiseKernel{T},
                X::AbstractMatrix{T}
            )
            if !(n = size(X, $dimension) == size(K,1) == size(K,2))
                throw(DimensionMismatch("Dimensions of K must match dimension $dimension of X"))
            end
            for j = 1:n
                xj = subvector(Val{$scheme}, X, j)
                for i = j:n
                    xi = subvector(Val{$scheme}, X, i)
                    X[i,j] = vectorpairwise(κ, xi, xj)
                end
            end
            LinAlg.copytri!(K, 'U', false)
        end

        function pairwise!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}}
                K::Matrix{T}, 
                κ::PairwiseKernel{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T},
            )
            if !(n = size(X, $dimension) == size(K,1))
                throw(DimensionMismatch("Dimension 1 of K must match dimension $dimension of X"))
            elseif !(m = size(Y, $dimension) == size(K,2))
                throw(DimensionMismatch("Dimension 2 of K must match dimension $dimension of Y"))
            end
            for j = 1:m
                yj = subvector(scheme, Y, j)
                for i = j:n
                    xi = subvector(scheme, X, i)
                    X[i,j] = vectorpairwise(κ, xi, yj)
                end
            end
            K
        end
    end
end

function pairwise{T<:AbstractFloat}(
        v::DataType
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwise!(v, init_pairwise(v, X, Y), κ, X, Y)
end

function pairwise{T<:AbstractFloat}(
        v::DataType
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwise!(v, init_pairwise(v, X, Y), κ, X, Y)
end


#===================================================================================================
  ScalarProduct and SquaredDistance using BLAS
===================================================================================================#

for (scheme, dimension) in ((:row, 1), (:col, 2))
    @eval begin

        function dotvectors!{T<:AbstractFloat}(
                ::Type{Val{$scheme}},
                X::AbstractMatrix{T}, 
                xᵀx::Vector{T}
            )
            if !(size(X,$dimension) == length(xᵀx))
                throw(DimensionMismatch("Dimension mismatch on dimension $dimension"))
            end
            zero!(xᵀx)
            for I in CartesianRange(size(X))
                xᵀx[I.I[$dimension]] += X[I]^2
            end
            xᵀx
        end

        @inline function dotvectors{T<:AbstractFloat}(::Type{Val{$scheme}}, X::AbstractMatrix{T})
            dotvectors!(X, Array(T, size(X,$dimension)))
        end

        @inline function gramian!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}}, 
                G::Matrix{T}, 
                X::Matrix{T}, 
                Y::Matrix{T}
            )
            $(scheme == :row ? :A_mul_Bt! : :At_mul_B!)(G, X, Y)
        end

        @inline function gramian!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}},
                G::Matrix{T},
                X::Matrix{T}
            )
            gramian(Val{$scheme}, G, X, X)
        end
    end
end

function squared_distance!{T<:AbstractFloat}(G::Matrix{T}, xᵀx::Vector{T})
    if !((n = length(xᵀx)) == size(K,1) == size(K,2))
        throw(DimensionMismatch("Gramian matrix must be square."))
    end
    @inbounds for j = 1:n, i = (j:n)
        G[i,j] = xᵀx[i] - 2K[i,j] + xᵀx[j]
    end
    LinAlg.copytri!(G, 'U')
end

function squared_distance!{T<:AbstractFloat}(G::Matrix{T}, xᵀx::Vector{T}, yᵀy::Vector{T})
    if size(G,1) != length(xᵀx)
        throw(DimensionMismatch(""))
    elseif size(G,2) != length(yᵀy)
        throw(DimensionMismatch(""))
    end
    @inbounds for I in CartesianRange(G)
        G[I] = xᵀx[I[1]] - 2G[I] + yᵀy[I[2]]
    end
    G
end

@inline function pairwise!{T<:AbstractFloat}(
        v::DataType
        K::Matrix{T}, 
        κ::ScalarProductKernel{T},
        X::AbstractMatrix{T}
    )
    gramian!(v, K, X)
end

@inline function pairwise!{T<:AbstractFloat}(
        v::DataType
        K::Matrix{T}, 
        κ::ScalarProductKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T},
    )
    gramian!(v, K, X)
end

@inline function pairwise!{T<:AbstractFloat}(
        v::DataType
        K::Matrix{T}, 
        κ::SquaredDistanceKernel{T},
        X::AbstractMatrix{T}
    )
    gramian!(v, K, X)
    xᵀx = dotvectors(v, X)
    squared_distance!(K, xᵀx)
end

@inline function pairwise!{T<:AbstractFloat}(
        v::DataType
        K::Matrix{T}, 
        κ::SquaredDistanceKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T},
    )
    gramian!(v, K, X, Y)
    xᵀx = dotvectors(v, X)
    yᵀy = dotvectors(v, Y)
    squared_distance!(K, xᵀx, yᵀy)
end
