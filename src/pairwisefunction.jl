#===================================================================================================
  Generic pairwisematrix functions for kernels consuming two vectors
===================================================================================================#

pairwise{T}(κ::StandardKernel{T}, x::T, y::T) = phi(κ, x, y)
pairwise{T}(κ::StandardKernel{T}, x::AbstractArray{T}, y::AbstractArray{T}) = phi(κ, x, y)

pairwise{T}(κ::AdditiveKernel{T}, x::T, y::T) = phi(κ, x, y)
function unsafe_pairwise{T}(κ::AdditiveKernel{T}, x::AbstractArray{T}, y::AbstractArray{T})
    s = zero(T)
    @inbounds for i in eachindex(x)
        s += phi(κ, x[i], y[i])
    end
    s
end
function pairwise{T}(κ::AdditiveKernel{T}, x::AbstractArray{T}, y::AbstractArray{T})
    length(x) == length(y) || throw(DimensionMismatch("Arrays x and y must have the same length."))
    unsafe_pairwise(κ, x, y)
end


for (order, dimension) in ((:(:row), 1), (:(:col), 2))
    isrowmajor = order == :(:row)
    @eval begin

        @inline function subvector(::Type{Val{$order}}, X::AbstractMatrix,  i::Integer)
            $(isrowmajor ? :(slice(X, i, :)) : :(slice(X, :, i)))
        end

        @inline function init_pairwisematrix{T}(
                 ::Type{Val{$order}},
                X::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(X,$dimension))
        end

        @inline function init_pairwisematrix{T}(
                 ::Type{Val{$order}},
                X::AbstractMatrix{T}, 
                Y::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(Y,$dimension))
        end

        function checkpairwisedimensions{T}(
                 ::Type{Val{$order}},
                K::Matrix{T}, 
                X::AbstractMatrix{T}
            )
            n = size(K,1)
            if size(K,2) != n
                throw(DimensionMismatch("Kernel matrix K must be square"))
            elseif size(X, $dimension) != n
                errorstring = string("Dimensions of K must match dimension ", $dimension, "of X")
                throw(DimensionMismatch(errorstring))
            end
            return n
        end

        function checkpairwisedimensions(
                 ::Type{Val{$order}},
                K::Matrix,
                X::AbstractMatrix, 
                Y::AbstractMatrix
            )
            n = size(X, $dimension)
            m = size(Y, $dimension)
            if n != size(K,1)
                errorstring = string("Dimension 1 of K must match dimension ", $dimension, "of X")
                throw(DimensionMismatch(errorstring))
            elseif m != size(K,2)
                errorstring = string("Dimension 2 of K must match dimension ", $dimension, "of Y")
                throw(DimensionMismatch(errorstring))
            end
            return (n, m)
        end

        function pairwisematrix!{T}(
                σ::Type{Val{$order}},
                K::Matrix{T}, 
                κ::PairwiseKernel{T},
                X::AbstractMatrix{T}
            )
            n = checkpairwisedimensions(σ, K, X)
            for j = 1:n
                xj = subvector(σ, X, j)
                for i = 1:j
                    xi = subvector(σ, X, i)
                    @inbounds K[i,j] = unsafe_pairwise(κ, xi, xj)
                end
            end
            LinAlg.copytri!(K, 'U', false)
        end

        function pairwisematrix!{T}(
                σ::Type{Val{$order}},
                K::Matrix{T}, 
                κ::PairwiseKernel{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T},
            )
            n, m = checkpairwisedimensions(σ, K, X, Y)
            for j = 1:m
                yj = subvector(σ, Y, j)
                for i = 1:n
                    xi = subvector(σ, X, i)
                    @inbounds K[i,j] = unsafe_pairwise(κ, xi, yj)
                end
            end
            K
        end
    end
end

function pairwisematrix{T}(
        σ::DataType,
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T}
    )
    pairwisematrix!(σ, init_pairwisematrix(σ, X), κ, X)
end

function pairwisematrix{T}(
        σ::DataType,
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(σ, init_pairwisematrix(σ, X, Y), κ, X, Y)
end


#===================================================================================================
  ScalarProduct and SquaredDistance using BLAS/Built-In methods
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
        throw(DimensionMismatch("Length of xᵀx must match rows of G"))
    elseif size(G,2) != length(yᵀy)
        throw(DimensionMismatch("Length of yᵀy must match columns of G"))
    end
    @inbounds for I in CartesianRange(size(G))
        G[I] = xᵀx[I[1]] - 2G[I] + yᵀy[I[2]]
    end
    G
end

for (order, dimension) in ((:(:row), 1), (:(:col), 2))
    isrowmajor = order == :(:row)
    @eval begin

        function dotvectors!{T<:AbstractFloat}(
                 ::Type{Val{$order}},
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

        @inline function dotvectors{T<:AbstractFloat}(σ::Type{Val{$order}}, X::AbstractMatrix{T})
            dotvectors!(σ, Array(T, size(X,$dimension)), X)
        end

        function gramian!{T<:AbstractFloat}(
                 ::Type{Val{$order}},
                G::Matrix{T},
                X::Matrix{T}
            )
            $(isrowmajor ? :A_mul_Bt! : :At_mul_B!)(G, X, X)
        end

        function gramian!{T<:AbstractFloat}(
                σ::Type{Val{$order}},
                G::Matrix{T},
                X::AbstractMatrix{T}
            )
            checkpairwisedimensions(σ, G, X)
            copy!(G, $(isrowmajor ? :A_mul_Bt : :At_mul_B)(X, X))
        end

        function gramian!{T<:AbstractFloat}(
                 ::Type{Val{$order}}, 
                G::Matrix{T}, 
                X::Matrix{T}, 
                Y::Matrix{T}
            )
            $(isrowmajor ? :A_mul_Bt! : :At_mul_B!)(G, X, Y)
        end

        function gramian!{T<:AbstractFloat}(
                σ::Type{Val{$order}}, 
                G::Matrix{T}, 
                X::AbstractMatrix{T}, 
                Y::AbstractMatrix{T}
            )
            checkpairwisedimensions(σ, G, X, Y)
            # copy!(G, $(isrowmajor ? :A_mul_Bt : :At_mul_B)(X, Y))
            copy!(G, $(isrowmajor ? :(X*transpose(Y)) : :(transpose(X)*Y)))
        end

        @inline function pairwisematrix!{T}(
                σ::Type{Val{$order}},
                K::Matrix{T}, 
                κ::ScalarProductKernel{T},
                X::AbstractMatrix{T}
            )
            gramian!(σ, K, X)
        end

        @inline function pairwisematrix!{T}(
                σ::Type{Val{$order}},
                K::Matrix{T}, 
                κ::ScalarProductKernel{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T},
            )
            gramian!(σ, K, X, Y)
        end

        @inline function pairwisematrix!{T}(
                σ::Type{Val{$order}},
                K::Matrix{T}, 
                κ::SquaredDistanceKernel{T},
                X::AbstractMatrix{T}
            )
            gramian!(σ, K, X)
            xᵀx = dotvectors(σ, X)
            squared_distance!(K, xᵀx)
        end

        @inline function pairwisematrix!{T}(
                σ::Type{Val{$order}},
                K::Matrix{T}, 
                κ::SquaredDistanceKernel{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T},
            )
            gramian!(σ, K, X, Y)
            xᵀx = dotvectors(σ, X)
            yᵀy = dotvectors(σ, Y)
            squared_distance!(K, xᵀx, yᵀy)
        end
    end
end
