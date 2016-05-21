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


for (scheme, dimension) in ((:(:row), 1), (:(:col), 2))
    isrowmajor = scheme == :(:row)
    @eval begin

        @inline function subvector(::Type{Val{$scheme}}, X::AbstractMatrix,  i::Integer)
            $(isrowmajor ? :(slice(X, i, :)) : :(slice(X, :, i)))
        end

        @inline function init_pairwisematrix{T}(
                 ::Type{Val{$scheme}},
                X::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(X,$dimension))
        end

        @inline function init_pairwisematrix{T}(
                 ::Type{Val{$scheme}},
                X::AbstractMatrix{T}, 
                Y::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(Y,$dimension))
        end

        function checkpairwisedimensions{T}(
                 ::Type{Val{$scheme}},
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
                 ::Type{Val{$scheme}},
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
                v::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::PairwiseKernel{T},
                X::AbstractMatrix{T}
            )
            n = checkpairwisedimensions(v, K, X)
            for j = 1:n
                xj = subvector(v, X, j)
                for i = 1:j
                    xi = subvector(v, X, i)
                    @inbounds K[i,j] = unsafe_pairwise(κ, xi, xj)
                end
            end
            LinAlg.copytri!(K, 'U', false)
        end

        function pairwisematrix!{T}(
                v::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::PairwiseKernel{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T},
            )
            n, m = checkpairwisedimensions(v, K, X, Y)
            for j = 1:m
                yj = subvector(v, Y, j)
                for i = 1:n
                    xi = subvector(v, X, i)
                    @inbounds K[i,j] = unsafe_pairwise(κ, xi, yj)
                end
            end
            K
        end
    end
end

function pairwisematrix{T}(
        v::DataType,
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T}
    )
    pairwisematrix!(v, init_pairwisematrix(v, X), κ, X)
end

function pairwisematrix{T}(
        v::DataType,
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(v, init_pairwisematrix(v, X, Y), κ, X, Y)
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

for (scheme, dimension) in ((:(:row), 1), (:(:col), 2))
    isrowmajor = scheme == :(:row)
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

        @inline function dotvectors{T<:AbstractFloat}(v::Type{Val{$scheme}}, X::AbstractMatrix{T})
            dotvectors!(v, Array(T, size(X,$dimension)), X)
        end

        function gramian!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}},
                G::Matrix{T},
                X::Matrix{T}
            )
            $(isrowmajor ? :A_mul_Bt! : :At_mul_B!)(G, X, X)
        end

        function gramian!{T<:AbstractFloat}(
                v::Type{Val{$scheme}},
                G::Matrix{T},
                X::AbstractMatrix{T}
            )
            checkpairwisedimensions(v, G, X)
            copy!(G, $(isrowmajor ? :A_mul_Bt : :At_mul_B)(X, X))
        end

        function gramian!{T<:AbstractFloat}(
                 ::Type{Val{$scheme}}, 
                G::Matrix{T}, 
                X::Matrix{T}, 
                Y::Matrix{T}
            )
            $(isrowmajor ? :A_mul_Bt! : :At_mul_B!)(G, X, Y)
        end

        function gramian!{T<:AbstractFloat}(
                v::Type{Val{$scheme}}, 
                G::Matrix{T}, 
                X::AbstractMatrix{T}, 
                Y::AbstractMatrix{T}
            )
            checkpairwisedimensions(v, G, X, Y)
            # copy!(G, $(isrowmajor ? :A_mul_Bt : :At_mul_B)(X, Y))
            copy!(G, $(isrowmajor ? :(X*transpose(Y)) : :(transpose(X)*Y)))
        end

        @inline function pairwisematrix!{T}(
                v::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::ScalarProductKernel{T},
                X::AbstractMatrix{T}
            )
            gramian!(v, K, X)
        end

        @inline function pairwisematrix!{T}(
                v::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::ScalarProductKernel{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T},
            )
            gramian!(v, K, X, Y)
        end

        @inline function pairwisematrix!{T}(
                v::Type{Val{$scheme}},
                K::Matrix{T}, 
                κ::SquaredDistanceKernel{T},
                X::AbstractMatrix{T}
            )
            gramian!(v, K, X)
            xᵀx = dotvectors(v, X)
            squared_distance!(K, xᵀx)
        end

        @inline function pairwisematrix!{T}(
                v::Type{Val{$scheme}},
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
    end
end
