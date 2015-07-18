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

# Apply kappa to matrix elements
function kappa_matrix!{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T})
    @inbounds @simd for i = 1:length(X)
        X[i] = kappa(κ, X[i])
    end
    X
end
kappa_matrix{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}) = kappa_array!(κ, copy(X))

function kappa_square_matrix!{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, store_upper::Bool)
    (n = size(X,1)) == size(X,2) || throw(DimensionMismatch("X must be square."))
    @inbounds for j = 1:n, i = store_upper ? (1:j) : (j:n)
        X[i,j] = kappa(κ, X[i,j])
    end
    X
end
kappa_square_matrix{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, store_upper::Bool) = kappa_array!(κ, copy(X), store_upper)


#===================================================================================================
  Default Pairwise Computation
===================================================================================================#

# Initiate pairwise matrices

function init_pairwise{T<:FloatingPoint}(X::Matrix{T}, is_trans::Bool)
    n = size(X, is_trans ? 2 : 1)
    Array(T, n, n)
end

function init_pairwise{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
    n_dim = is_trans ? 2 : 1
    n = size(X, n_dim)
    m = size(Y, n_dim)
    Array(T, n, m)
end


# Pairwise definition

function pairwise!{T<:FloatingPoint}(K::Matrix{T}, κ::BaseKernel{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool)
    if is_trans
        pairwise_Xt!(K, κ, X, store_upper)
    else
        pairwise_X!(K, κ, X, store_upper)
    end
end
function pairwise{T<:FloatingPoint}(κ::BaseKernel{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool)
    pairwise!(init_pairwise(X, is_trans), κ, X, is_trans, store_upper)
end

function pairwise!{T<:FloatingPoint}(K::Matrix{T}, κ::BaseKernel{T}, X::Matrix{T}, w::Vector{T}, is_trans::Bool, store_upper::Bool)
    if is_trans
        pairwise_Xt!(K, κ, X, w, store_upper)
    else
        pairwise_X!(K, κ, X, w, store_upper)
    end
end
function pairwise{T<:FloatingPoint}(κ::BaseKernel{T}, X::Matrix{T}, w::Vector{T}, is_trans::Bool, store_upper::Bool)
    pairwise!(init_pairwise(X, is_trans), κ, X, w, is_trans, store_upper)
end


# Default Pairwise Calculation

for (fn, dim_n, X_i, X_j, Y_j) in (
        (:pairwise_X!,  1, parse("X[i,:]"), parse("X[j,:]"), parse("Y[j,:]")),
        (:pairwise_Xt!, 2, parse("X[:,i]"), parse("X[:,j]"), parse("Y[:,j]"))
    )
    dim_p = dim_n == 1 ? 2 : 1
    @eval begin

        function ($fn){T<:FloatingPoint}(K::Matrix{T}, κ::BaseKernel{T}, X::Matrix{T}, store_upper::Bool)
            (n = size(X,$dim_n)) == size(K,1) == size(K,2) || throw(DimensionMismatch("Kernel matrix must be square and match X."))
            p = size(X,$dim_p)
            for j = 1:n, i = store_upper ? (1:j) : (j:n)
                K[i,j] = kappa(κ, vec($X_i), vec($X_j))
            end
            K
        end

        function ($fn){T<:FloatingPoint}(K::Matrix{T}, κ::AdditiveKernel{T}, X::Matrix{T}, Y::Matrix{T})
            (n = size(X,$dim_n)) == size(K,1) || throw(DimensionMismatch("Dimension $($dim_n) of X must match dimension 1 of K."))
            (m = size(Y,$dim_n)) == size(K,2) || throw(DimensionMismatch("Dimension $($dim_n) of Y must match dimension 2 of K."))
            size(X,$dim_p) == size(Y,$dim_p) || throw(DimensionMismatch("Dimension $($dim_p) of Y must match dimension $($dim_p) of X."))
            for j = 1:m, i = 1:n
                K[i,j] = kappa(κ, vec($X_i), vec($Y_j))
            end
            K
        end

    end

end


#===========================================================================
  "Optimised" Generic Additive Pairwise 
===========================================================================#

pairwise{T<:FloatingPoint}(κ::AdditiveKernel{T}, x::T, y::T) = kappa(κ, x, y)
pairwise{T<:FloatingPoint}(κ::AdditiveKernel{T}, x::T, y::T, w::T) = w * w * kappa(κ, x, y)

function pairwise{T<:FloatingPoint}(κ::AdditiveKernel{T}, x::Vector{T}, y::Vector{T})
    (n = length(x)) == length(y) || throw(DimensionMismatch("x and y must be of the same dimension."))
    v = zero(T)
    @inbounds @simd for i = 1:n
        v += kappa(κ, x[i], y[i])
    end
    v
end

function pairwise{T<:FloatingPoint}(κ::AdditiveKernel{T}, x::Vector{T}, y::Vector{T}, w::Vector{T})
    (n = length(x)) == length(y) || throw(DimensionMismatch("x and y must be the same dimension."))
    n == length(w) || throw(DimensionMismatch("w must have the same dimension as x and y."))
    v = zero(T)
    @inbounds @simd for i = 1:n
        w² = w[i] * w[i]
        v += w² * kappa(κ, x[i], y[i])
    end
    v
end

for (fn_X, fn_XY, dim_n, X_ji, X_ki, Y_ki) in (
        (:pairwise_X!,  :pairwise_XY,   1, parse("X[j,i]"), parse("X[k,i]"), parse("Y[k,i]")),
        (:pairwise_Xt!, :pairwise_XtYt, 2, parse("X[i,j]"), parse("X[i,k]"), parse("Y[i,k]"))
    )
    dim_p = dim_n == 1 ? 2 : 1
    @eval begin

        function ($fn_X){T<:FloatingPoint}(K::Matrix{T}, κ::AdditiveKernel{T}, X::Matrix{T}, store_upper::Bool)
            (n = size(X,$dim_n)) == size(K,1) == size(K,2) || throw(DimensionMismatch("Kernel matrix must be square and match X."))
            p = size(X,$dim_p)
            for k = 1:n, j = store_upper ? (1:k) : (k:n)
                v = 0
                @inbounds @simd for i = 1:p
                    v += kappa(κ, $X_ji, $X_ki)
                end
                K[j,k] = v
            end
            K
        end

        function ($fn_X){T<:FloatingPoint}(K::Matrix{T}, κ::AdditiveKernel{T}, X::Matrix{T}, w::Vector{T}, store_upper::Bool)
            (n = size(X,$dim_n)) == size(K,1) == size(K,2) || throw(DimensionMismatch("Kernel matrix must be square and match X."))
            (p = size(X,$dim_p)) == length(w) || throw(DimensionMismatch("Weight vector w must match X."))
            w² = w.^2
            for k = 1:n, j = store_upper ? (1:k) : (k:n)
                v = 0
                @inbounds @simd for i = 1:p
                    v += w²[i] * kappa(κ, $X_ji, $X_ki)
                end
                K[j,k] = v
            end
            K
        end

        function ($fn_XY){T<:FloatingPoint}(K::Matrix{T}, κ::AdditiveKernel{T}, X::Matrix{T}, Y::Matrix{T})
            (n = size(X,$dim_n)) == size(K,1) || throw(DimensionMismatch("Dimension $($dim_n) of X must match dimension 1 of K."))
            (m = size(Y,$dim_n)) == size(K,2) || throw(DimensionMismatch("Dimension $($dim_n) of Y must match dimension 2 of K."))
            (p = size(X,$dim_p)) == size(Y,$dim_p) || throw(DimensionMismatch("Dimension $($dim_p) of X must match $($dim_p) of Y."))
            for k = 1:m, j = 1:n
                v = 0
                @inbounds @simd for i = 1:p
                    v += kappa(κ, $X_ji, $Y_ki)
                end
                K[j,k] = v
            end
            K
        end

        function ($fn_XY){T<:FloatingPoint}(K::Matrix{T}, κ::AdditiveKernel{T}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T})
            (n = size(X,$dim_n)) == size(K,1) || throw(DimensionMismatch("Dimension $($dim_n) of X must match dimension 1 of K."))
            (m = size(Y,$dim_n)) == size(K,2) || throw(DimensionMismatch("Dimension $($dim_n) of Y must match dimension 2 of K."))
            (p = size(X,$dim_p)) == size(Y,$dim_p) || throw(DimensionMismatch("Dimension $($dim_p) of X must match $($dim_p) of Y."))
            p == length(w) || throw(DimensionMismatch("Length of w must match dimension $($dim_p) of X and Y."))
            w² = w.^2
            for k = 1:m, j = 1:n
                v = 0
                @inbounds @simd for i = 1:p
                    v += w²[i] * kappa(κ, $X_ji, $Y_ki)
                end
                K[j,k] = v
            end
            K
        end

    end
end

#  ARD - Automatic Relevance Determination

pairwise!{T<:FloatingPoint}(K::Matrix{T}, κ::ARD{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool) = pairwise!(K, κ.k, X, κ.w, is_trans, store_upper)
pairwise{T<:FloatingPoint}(κ::ARD{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool) = pairwise(κ.k, X, κ.w, is_trans, store_upper)

pairwise!{T<:FloatingPoint}(K::Matrix{T}, κ::ARD{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool) = pairwise!(K, κ.k, X, Y, κ.w, is_trans)
pairwise{T<:FloatingPoint}(κ::ARD{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool) = pairwise(κ.k, X, Y, κ.w, is_trans)


#===========================================================================
  Optimized Separable Kernel
===========================================================================#

# Base Scalar Product

pairwise_X!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, store_upper::Bool) = gramian_X!(K, X, store_upper)
pairwise_Xt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, store_upper::Bool) = gramian_Xt!(K, X, store_upper)

pairwise_X!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, w::Vector{T}, store_upper::Bool) = gramian_X!(K, scale(X, w), store_upper)
pairwise_Xt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, w::Vector{T}, store_upper::Bool) = gramian_Xt!(K, scale(w, X), store_upper)

pairwise_XY!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, Y::Matrix{T}) = gramian_XY!(K, X, Y)
pairwise_XtYt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, Y::Matrix{T}) = gramian_XtYt!(K, X, Y)

pairwise_XY!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T}) = gramian_XY!(K, scale(X, w.^2), Y)
pairwise_XtYt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::ScalarProductKernel{T}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T}) = gramian_XtYt!(K, scale(w.^2, X), Y)

# Separable Kernel

function pairwise_X!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SeparableKernel{T}, X::Matrix{T}, store_upper::Bool)
    Z = kappa_matrix(κ, X)
    gramian_X!(K, Z, store_upper)
end
function pairwise_Xt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SeparableKernel{T}, X::Matrix{T}, store_upper::Bool)
    Z = kappa_matrix(κ, X)
    gramian_Xt!(K, Z, store_upper)
end

function pairwise_X!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SeparableKernel{T}, X::Matrix{T}, w::Vector{T}, store_upper::Bool)
    Z = scale!(kappa_matrix(κ, X), w)
    gramian_X!(K, Z, store_upper)
end
function pairwise_Xt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SeparableKernel{T}, X::Matrix{T}, w::Vector{T}, store_upper::Bool)
    Z = scale!(kappa_matrix(X, κ), w)
    gramian_Xt!(K, Z, store_upper)
end

function pairwise_X!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SeparableKernel{T}, X::Matrix{T}, Y::Matrix{T})
    Z = kappa_matrix(κ, X)
    V = kappa_matrix(κ, Y)
    gramian_X!(K, Z, V)
end
function pairwise_Xt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SeparableKernel{T}, X::Matrix{T}, Y::Matrix{T})
    Z = kappa_matrix(κ, X)
    V = kappa_matrix(κ, Y)
    gramian_Xt!(K, Z, V)
end

function pairwise_XY!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SeparableKernel{T}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T})
    Z = scale!(kappa_matrix(κ, X), w)
    V = scale!(kappa_matrix(κ, Y), w)
    gramian_X!(K, Z, V)
end
function pairwise_XtYt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SeparableKernel{T}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T})
    Z = scale!(kappa_matrix(X, κ), w)
    V = scale!(kappa_matrix(κ, Y), w)
    gramian_Xt!(K, Z, V)
end


#===========================================================================
  Optimized Squared Distance Kernel
===========================================================================#

function squared_distance!{T<:FloatingPoint}(K::Matrix{T}, xᵀx::Vector{T}, store_upper::Bool)
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

function squared_distance!{T<:FloatingPoint}(K::Matrix{T}, xᵀx::Vector{T}, yᵀy::Vector{T})
    n,m = size(K)
    n == length(xᵀx) || throw(DimensionMismatch(""))
    m == length(yᵀy) || throw(DimensionMismatch(""))
    @inbounds for j = 1:m, i = 1:n
        K[i,j] = xᵀx[i] - 2K[i,j] + yᵀy[j]
    end
    K
end

function pairwise_XY!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, Y::Matrix{T}, store_upper::Bool)
    gramian_X!(K, X, Y)
    squared_distance!(K, dot_rows(X), dot_rows(Y))
end

function pairwise_XtYt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, Y::Matrix{T}, store_upper::Bool)
    gramian_Xt!(K, X, Y)
    squared_distance!(K, dot_columns(X), dot_columns(Y))
end

function pairwise_XY!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T}, store_upper::Bool)
    Z = scale(X, w)
    V = scale(Y, w)
    gramian_X!(K, Z, V)
    squared_distance!(K, dot_rows(Z), dot_rows(V))
end

function pairwise_XtYt!{T<:Base.LinAlg.BlasReal}(K::Matrix{T}, κ::SquaredDistanceKernel{T,:t1}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T}, store_upper::Bool)
    Z = scale(w, X)
    V = scale(w, Y)
    gramian_Xt!(K, X, Y)
    squared_distance!(K, dot_columns(Z), dot_columns(V))
end
