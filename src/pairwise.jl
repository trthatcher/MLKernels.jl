# Helper Functions

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

        @inline function init_pairwise{T<:AbstractFloat}(::Type{Val{$scheme}}, X::AbstractMatrix{T})
            Array(T, size(X,$dimension), size(X,$dimension))
        end

        @inline function init_pairwise{T<:AbstractFloat}(
                 ::Type{Val{$scheme}},
                X::AbstractMatrix{T}, 
                Y::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(Y,$dimension))
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

        @inline function subvector(::Type{Val{$scheme}}, X::AbstractMatrix,  i::Integer)
            $(scheme == :row ? :(sub(X, i, :)) : :(sub(X, :, i)))
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


function pairwise!{T<:AbstractFloat}(
        K::Matrix{T}, 
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T},
        is_rowmajor::Bool
    )
    dim, scheme = is_rowmajor ? (1, Val{:row}) : (2, Val{:Col})
    if !(n = size(X, dim) == size(K,1) == size(K,2))
        throw(DimensionMismatch("Dimensions of K must match dimension $dim of X"))
    end
    for j = 1:n
        xj = subvector(scheme, X, j)
        for i = j:n
            xi = subvector(scheme, X, i)
            X[i,j] = vectorpairwise(κ, xi, xj)
        end
    end
    LinAlg.copytri!(K, 'U', false)
end

function pairwise!{T<:AbstractFloat}(
        K::Matrix{T}, 
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T},
        is_rowmajor::Bool
    )
    dim, scheme = is_rowmajor ? (1, Val{:row}) : (2, Val{:Col})
    if !(n = size(X, dim) == size(K,1))
        throw(DimensionMismatch("Dimension 1 of K must match dimension $dim of X"))
    elseif !(m = size(Y, dim) == size(K,2))
        throw(DimensionMismatch("Dimension 2 of K must match dimension $dim of Y"))
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
function pairwise{T<:AbstractFloat}(
        κ::PairwiseKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T},
        is_rowmajor::Bool
    )
    pairwise!(init_gramian

# NO CHECKS
@inline function vectorpairwise{T<:AbstractFloat}(
        κ::StandardKernel{T},
        x::AbstractArray{T}, 
        y::AbstractArray{T}
    )
    phi(κ, x, y)
end

# NO CHECKS!
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

















#===================================================================================================
  Pairwise Computation
===================================================================================================#

# Calculate the gramian matrix of X
function gramian_X!{T<:Base.LinAlg.BlasReal}(G::Matrix{T}, X::Matrix{T}, store_upper::Bool)
    (n = size(G, 1)) == size(G, 2) == size(X, 1) || throw(DimensionMismatch("Supplied kernel matrix must be square and have same number of rows as X."))
    BLAS.syrk!(store_upper ? 'U' : 'L', 'N', one(T), X, zero(T), G)
end

function gramian_Xt!{T<:Base.LinAlg.BlasReal}(G::Matrix{T}, X::Matrix{T}, store_upper::Bool)
    (n = size(G, 1)) == size(G, 2) == size(X, 2) || throw(DimensionMismatch("Supplied kernel matrix must be square and have the same number of colums as X."))
    BLAS.syrk!(store_upper ? 'U' : 'L', 'T', one(T), X, zero(T), G)  # 'T' -> C := αA'A + βC
end

# Returns the upper right corner of the gramian of [X Y] or [Xᵀ Yᵀ]ᵀ
function gramian_XY!{T<:Base.LinAlg.BlasReal}(G::Matrix{T}, X::Matrix{T}, Y::Matrix{T})
    size(X, 2) == size(Y, 2) || throw(DimensionMismatch("X must have as many columns as Y."))
    size(X, 1) == size(G, 1) || throw(DimensionMismatch("Supplied kernel matrix must have as many rows as X has rows."))
    size(Y, 1) == size(G, 2) || throw(DimensionMismatch("Supplied kernel matrix must have as many columns as Y has rows."))
    BLAS.gemm!('N', 'T', one(T), X, Y, zero(T), G)
end

function gramian_XtYt!{T<:Base.LinAlg.BlasReal}(G::Matrix{T}, X::Matrix{T}, Y::Matrix{T})
    size(X, 1) == size(Y, 1) || throw(DimensionMismatch("X must have as many rows as Y."))
    size(X, 2) == size(G, 1) || throw(DimensionMismatch("Supplied kernel matrix must have as many rows as X has columns."))
    size(Y, 2) == size(G, 2) || throw(DimensionMismatch("Supplied kernel matrix must have as many columns as Y has columns."))
    BLAS.gemm!('T', 'N', one(T), X, Y, zero(T), G)
end


#===================================================================================================
  Default Pairwise Computation
===================================================================================================#

# Initiate pairwise matrices


# Pairwise definition

function pairwise!{T<:AbstractFloat}(K::Matrix{T}, κ::BaseKernel{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool)
    is_trans ? pairwise_Xt!(K, κ, X, store_upper) : pairwise_X!(K, κ, X, store_upper)
end
function pairwise{T<:AbstractFloat}(κ::BaseKernel{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool)
    pairwise!(init_pairwise(X, is_trans), κ, X, is_trans, store_upper)
end

function pairwise!{T<:AbstractFloat}(K::Matrix{T}, κ::BaseKernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
    is_trans ? pairwise_XtYt!(K, κ, X, Y) : pairwise_XY!(K, κ, X, Y)
end
function pairwise{T<:AbstractFloat}(κ::BaseKernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
    pairwise!(init_pairwise(X, Y, is_trans), κ, X, Y, is_trans)
end


# Default Pairwise Calculation

pairwise{T<:AbstractFloat}(κ::BaseKernel{T}, x::T, y::T) = phi(κ, [x], [y])
pairwise{T<:AbstractFloat}(κ::BaseKernel{T}, x::Vector{T}, y::Vector{T}) = phi(κ, x, y)

@inline function subvector_X(X::AbstractMatrix,  i::Int64, n::Int64 = size(X,1), p::Int64 = size(X,2))
    sub(X, i:n:((p-1)*n + i))
end

@inline function subvector_Xt(X::AbstractMatrix, i::Int64, n::Int64 = size(X,2), p::Int64 = size(X,1))
    sub(X,((i-1)*p + 1):1:(i*p))
end

for (fn_X, fn_XY, fn_subvec, dim_n) in (
        (:pairwise_X!,  :pairwise_XY!,   :subvector_X,  1),
        (:pairwise_Xt!, :pairwise_XtYt!, :subvector_Xt, 2)
    )
    dim_p = dim_n == 1 ? 2 : 1
    @eval begin

        function ($fn_X){T<:AbstractFloat}(K::Matrix{T}, κ::BaseKernel{T}, X::Matrix{T}, store_upper::Bool)
            (n = size(X,$dim_n)) == size(K,1) == size(K,2) || throw(DimensionMismatch("Kernel matrix must be square and match X."))
            p = size(X,$dim_p)
            for j = 1:n
                y = ($fn_subvec)(X,j)
                for i = store_upper ? (1:j) : (j:n)
                    K[i,j] = phi(κ, ($fn_subvec)(X,i), y)
                end
            end
            K
        end

        function ($fn_XY){T<:AbstractFloat}(K::Matrix{T}, κ::BaseKernel{T}, X::Matrix{T}, Y::Matrix{T})
            (n = size(X,$dim_n)) == size(K,1) || throw(DimensionMismatch("Dimension $($dim_n) of X must match dimension 1 of K."))
            (m = size(Y,$dim_n)) == size(K,2) || throw(DimensionMismatch("Dimension $($dim_n) of Y must match dimension 2 of K."))
            size(X,$dim_p) == size(Y,$dim_p) || throw(DimensionMismatch("Dimension $($dim_p) of Y must match dimension $($dim_p) of X."))
            for j = 1:m
                y = ($fn_subvec)(Y,j)
                for i = 1:n
                    K[i,j] = phi(κ, ($fn_subvec)(X,i), y)
                end
            end
            K
        end

    end

end


#===========================================================================
  "Optimised" Generic Additive Pairwise 
===========================================================================#

pairwise{T<:AbstractFloat}(κ::AdditiveKernel{T}, x::T, y::T) = phi(κ, x, y)
pairwise{T<:AbstractFloat}(κ::AdditiveKernel{T}, x::T, y::T, w::T) = w * w * phi(κ, x, y)

function pairwise{T<:AbstractFloat}(κ::AdditiveKernel{T}, x::Vector{T}, y::Vector{T})
    (n = length(x)) == length(y) || throw(DimensionMismatch("x and y must be of the same dimension."))
    v = zero(T)
    @inbounds @simd for i = 1:n
        v += phi(κ, x[i], y[i])
    end
    v
end

function pairwise{T<:AbstractFloat}(κ::AdditiveKernel{T}, x::Vector{T}, y::Vector{T}, w::Vector{T})
    (n = length(x)) == length(y) || throw(DimensionMismatch("x and y must be the same dimension."))
    n == length(w) || throw(DimensionMismatch("w must have the same dimension as x and y."))
    v = zero(T)
    @inbounds @simd for i = 1:n
        w² = w[i] * w[i]
        v += w² * phi(κ, x[i], y[i])
    end
    v
end

for (fn_X, fn_XY, dim_n, X_ji, X_ki, Y_ki) in (
        (:pairwise_X!,  :pairwise_XY!,   1, parse("X[j,i]"), parse("X[k,i]"), parse("Y[k,i]")),
        (:pairwise_Xt!, :pairwise_XtYt!, 2, parse("X[i,j]"), parse("X[i,k]"), parse("Y[i,k]"))
    )
    dim_p = dim_n == 1 ? 2 : 1
    @eval begin

        function ($fn_X){T<:AbstractFloat}(K::Matrix{T}, κ::AdditiveKernel{T}, X::Matrix{T}, store_upper::Bool)
            (n = size(X,$dim_n)) == size(K,1) == size(K,2) || throw(DimensionMismatch("Kernel matrix must be square and match X."))
            p = size(X,$dim_p)
            for k = 1:n, j = store_upper ? (1:k) : (k:n)
                v = 0
                @inbounds @simd for i = 1:p
                    v += phi(κ, $X_ji, $X_ki)
                end
                K[j,k] = v
            end
            K
        end

        function ($fn_X){T<:AbstractFloat}(K::Matrix{T}, κ::AdditiveKernel{T}, X::Matrix{T}, w::Vector{T}, store_upper::Bool)
            (n = size(X,$dim_n)) == size(K,1) == size(K,2) || throw(DimensionMismatch("Kernel matrix must be square and match X."))
            (p = size(X,$dim_p)) == length(w) || throw(DimensionMismatch("Weight vector w must match X."))
            w² = w.^2
            for k = 1:n, j = store_upper ? (1:k) : (k:n)
                v = 0
                @inbounds @simd for i = 1:p
                    v += w²[i] * phi(κ, $X_ji, $X_ki)
                end
                K[j,k] = v
            end
            K
        end

        function ($fn_XY){T<:AbstractFloat}(K::Matrix{T}, κ::AdditiveKernel{T}, X::Matrix{T}, Y::Matrix{T})
            (n = size(X,$dim_n)) == size(K,1) || throw(DimensionMismatch("Dimension $($dim_n) of X must match dimension 1 of K."))
            (m = size(Y,$dim_n)) == size(K,2) || throw(DimensionMismatch("Dimension $($dim_n) of Y must match dimension 2 of K."))
            (p = size(X,$dim_p)) == size(Y,$dim_p) || throw(DimensionMismatch("Dimension $($dim_p) of X must match $($dim_p) of Y."))
            for k = 1:m, j = 1:n
                v = 0
                @inbounds @simd for i = 1:p
                    v += phi(κ, $X_ji, $Y_ki)
                end
                K[j,k] = v
            end
            K
        end

        function ($fn_XY){T<:AbstractFloat}(K::Matrix{T}, κ::AdditiveKernel{T}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T})
            (n = size(X,$dim_n)) == size(K,1) || throw(DimensionMismatch("Dimension $($dim_n) of X must match dimension 1 of K."))
            (m = size(Y,$dim_n)) == size(K,2) || throw(DimensionMismatch("Dimension $($dim_n) of Y must match dimension 2 of K."))
            (p = size(X,$dim_p)) == size(Y,$dim_p) || throw(DimensionMismatch("Dimension $($dim_p) of X must match $($dim_p) of Y."))
            p == length(w) || throw(DimensionMismatch("Length of w must match dimension $($dim_p) of X and Y."))
            w² = w.^2
            for k = 1:m, j = 1:n
                v = 0
                @inbounds @simd for i = 1:p
                    v += w²[i] * phi(κ, $X_ji, $Y_ki)
                end
                K[j,k] = v
            end
            K
        end

    end
end

#  ARD - Automatic Relevance Determination

function pairwise{T<:AbstractFloat}(κ::ARD{T}, x::T, y::T) 
    length(κ.w) == 1 || throw(DimensionMismatch("w must be of length 1 to operate on scalars."))
    pairwise(κ.k, x, y, κ.w[1])
end

pairwise{T<:AbstractFloat}(κ::ARD{T}, x::Vector{T}, y::Vector{T}) = pairwise(κ.k, x, y, κ.w)

pairwise_X!{T<:AbstractFloat}(K::Matrix{T}, κ::ARD{T}, X::Matrix{T}, store_upper::Bool) = pairwise_X!(K, κ.k, X, κ.w, store_upper)
pairwise_Xt!{T<:AbstractFloat}(K::Matrix{T}, κ::ARD{T}, X::Matrix{T}, store_upper::Bool) = pairwise_Xt!(K, κ.k, X, κ.w, store_upper)

pairwise_XY!{T<:AbstractFloat}(K::Matrix{T}, κ::ARD{T}, X::Matrix{T}, Y::Matrix{T}) = pairwise_XY!(K, κ.k, X, Y, κ.w)
pairwise_XtYt!{T<:AbstractFloat}(K::Matrix{T}, κ::ARD{T}, X::Matrix{T}, Y::Matrix{T}) = pairwise_XtYt!(K, κ.k, X, Y, κ.w)


#===========================================================================
  Optimized Scalar Product
===========================================================================#

pairwise_X!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, store_upper::Bool) = gramian_X!(K, X, store_upper)
pairwise_Xt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, store_upper::Bool) = gramian_Xt!(K, X, store_upper)

pairwise_X!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, w::Vector{T}, store_upper::Bool) = gramian_X!(K, scale(X, w), store_upper)
pairwise_Xt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, w::Vector{T}, store_upper::Bool) = gramian_Xt!(K, scale(w, X), store_upper)

pairwise_XY!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, Y::Matrix{T}) = gramian_XY!(K, X, Y)
pairwise_XtYt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, Y::Matrix{T}) = gramian_XtYt!(K, X, Y)

pairwise_XY!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T}) = gramian_XY!(K, scale(X, w.^2), Y)
pairwise_XtYt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T}) = gramian_XtYt!(K, scale(w.^2, X), Y)


#===========================================================================
  Optimized Squared Distance Kernel
===========================================================================#

function squared_distance!{T<:AbstractFloat}(K::Matrix{T}, xᵀx::Vector{T}, store_upper::Bool)
    (n = length(xᵀx)) == size(K,1) == size(K,2) || throw(DimensionMismatch("Kernel matrix must be square."))
    @inbounds for j = 1:n, i = store_upper ? (1:j) : (j:n)
        K[i,j] = xᵀx[i] - 2K[i,j] + xᵀx[j]
    end
    K
end

function pairwise_X!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, store_upper::Bool)
    gramian_X!(K, X, store_upper)
    squared_distance!(K, diag(K), store_upper)
end

function pairwise_Xt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, store_upper::Bool)
    gramian_Xt!(K, X, store_upper)
    squared_distance!(K, diag(K), store_upper)
end

function pairwise_X!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, w::Vector{T}, store_upper::Bool)
    gramian_X!(K, scale(X, w), store_upper)
    squared_distance!(K, diag(K), store_upper)
end

function pairwise_Xt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, w::Vector{T}, store_upper::Bool)
    gramian_Xt!(K, scale(w, X), store_upper)
    squared_distance!(K, diag(K), store_upper)
end

function squared_distance!{T<:AbstractFloat}(K::Matrix{T}, xᵀx::Vector{T}, yᵀy::Vector{T})
    n,m = size(K)
    n == length(xᵀx) || throw(DimensionMismatch(""))
    m == length(yᵀy) || throw(DimensionMismatch(""))
    @inbounds for j = 1:m, i = 1:n
        K[i,j] = xᵀx[i] - 2K[i,j] + yᵀy[j]
    end
    K
end

function pairwise_XY!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, Y::Matrix{T})
    gramian_XY!(K, X, Y)
    squared_distance!(K, dot_rows(X), dot_rows(Y))
end

function pairwise_XtYt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, Y::Matrix{T})
    gramian_XtYt!(K, X, Y)
    squared_distance!(K, dot_columns(X), dot_columns(Y))
end

function pairwise_XY!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T})
    Z = scale(X, w)
    V = scale(Y, w)
    gramian_XY!(K, Z, V)
    squared_distance!(K, dot_rows(Z), dot_rows(V))
end

function pairwise_XtYt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T})
    Z = scale(w, X)
    V = scale(w, Y)
    gramian_XtYt!(K, Z, V)
    squared_distance!(K, dot_columns(Z), dot_columns(V))
end
