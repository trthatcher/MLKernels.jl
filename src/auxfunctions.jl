#===================================================================================================
  Auxiliary Functions
===================================================================================================#

description_matrix_size(A::Matrix) = string(size(A,1), "×", size(A,2))

# Symmetrize the lower half of matrix S using the upper half of S
function syml!(S::Matrix)
    (p = size(S,1)) == size(S,2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
    if p > 1 
        @inbounds for j = 1:(p - 1), i = (j + 1):p 
            S[i, j] = S[j, i]
        end
    end
    S
end
syml(S::Matrix) = syml!(copy(S))

# Symmetrize the upper off-diagonal of matrix S using the lower half of S
function symu!(S::Matrix)
    (p = size(S,1)) == size(S,2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
    if p > 1 
        @inbounds for j = 2:p, i = 1:(j-1)
            S[i,j] = S[j,i]
        end
    end
    S
end
symu(S::Matrix) = symu!(copy(S))

# Return vector of dot products for each row of A
function dot_rows{T<:FloatingPoint}(A::Matrix{T})
    n, m = size(A)
    aᵀa = zeros(T, n)
    @inbounds for j = 1:m, i = 1:n
        aᵀa[i] += A[i,j] * A[i,j]
    end
    aᵀa
end

# Return vector of dot products for each row of A
function dot_rows{T<:FloatingPoint}(A::Matrix{T}, w::Array{T})
    n, m = size(A)
    length(w) == m || throw(ArgumentError("w must have the same length as A's rows."))
    aᵀa = zeros(T, n)
    @inbounds for j = 1:m, i = 1:n
        aᵀa[i] += A[i,j] * A[i,j] * w[j]
    end
    aᵀa
end

# Return vector of dot products for each column of A
function dot_columns{T<:FloatingPoint}(A::Matrix{T})
    n, m = size(A)
    aᵀa = zeros(T, m)
    @inbounds for j = 1:m, i = 1:n
        aᵀa[j] += A[i,j] * A[i,j]
    end
    aᵀa
end

# Return vector of dot products for each column of A
function dot_columns{T<:FloatingPoint}(A::Matrix{T}, w::Array{T})
    n, m = size(A)
    length(w) == n || throw(ArgumentError("w must have the same length as A's rows."))
    aᵀa = zeros(T, m)
    @inbounds for j = 1:m, i = 1:n
        aᵀa[j] += A[i,j] * A[i,j] * w[i]
    end
    aᵀa
end


#==========================================================================
  Matrix Operations
==========================================================================#

# Overwrite A with the hadamard product of A and B. Returns A
function matrix_prod!{T<:FloatingPoint}(A::Array{T}, B::Array{T})
    length(A) == length(B) || error("A and B must be of the same length.")
    @inbounds for i = 1:length(A)
        A[i] *= B[i]
    end
    A
end

# Overwrite A with the hadamard product of A and B. Returns A
function matrix_prod!{T<:FloatingPoint}(A::Matrix{T}, B::Matrix{T}, is_upper::Bool, sym::Bool = true)
    (n = size(A,1)) == size(A,2) == size(B,1) == size(B,2) || throw(DimensionMismatch("A and B must be square and of same order."))
    if is_upper
        @inbounds for j = 1:n, i = 1:j
            A[i,j] *= B[i,j]
        end
        sym ? syml!(A) : A
    else
        @inbounds for j = 1:n, i = j:n
            A[i,j] *= B[i,j]
        end
        sym ? symu!(A) : A
    end
end

# Overwrite A with the matrix sum of A and B. Returns A
function matrix_sum!{T<:FloatingPoint}(A::Array{T}, B::Array{T})
    length(A) == length(B) || error("A and B must be of the same length.")
    @inbounds for i = 1:length(A)
        A[i] += B[i]
    end
    A
end

# Overwrite A with the matrix sum of A and B. Returns A
function matrix_sum!{T<:FloatingPoint}(A::Matrix{T}, B::Matrix{T}, is_upper::Bool, sym::Bool = true)
    (n = size(A,1)) == size(A,2) == size(B,1) == size(B,2) || throw(ArgumentError("A and B must be square and of same order."))
    if is_upper
        @inbounds for j = 1:n, i = 1:j
            A[i,j] += B[i,j]
        end
        sym ? syml!(A) : A
    else
        @inbounds for j = 1:n, i = j:n
            A[i,j] += B[i,j]
        end
        sym ? symu!(A) : A
    end
end

function translate!{T<:FloatingPoint}(A::Matrix{T}, b::T)
    @inbounds for i = 1:length(A)
        A[i] += b
    end
    A
end
translate!{T<:FloatingPoint}(b::T, A::Matrix{T}) = translate!(A, b)

#==========================================================================
  Vector Functions
==========================================================================#

# Scalar product of vectors x and y
function scprod{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    (n = length(x)) == length(y) || throw(ArgumentError("Dimensions do not conform."))
    z = zero(T)
    @inbounds @simd for i = 1:n
        z += x[i]*y[i]
    end
    z
end
scprod{T<:FloatingPoint}(x::T, y::T) =x*y

#=
# In-place scalar product calculation
function scprod!{T<:FloatingPoint}(d::Int64, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    z = zero(T)
    @transpose_access is_trans (X,Y) @inbounds for i = 1:d
        z += X[x_pos,i] * Y[y_pos,i]
    end
    z
end
=#

# Weighted scalar product of x and y
function scprod{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    z = zero(T)
    @inbounds @simd for i = 1:n
        z += x[i] * y[i] * w[i]^2
    end
    z
end
scprod{T<:FloatingPoint}(x::T, y::T, w::T) = x*y*w^2

#=
# In-place weighted scalar product calculation
function scprod!{T<:FloatingPoint}(d::Int64, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, w::Array{T}, is_trans::Bool)
    z = zero(T)
    @transpose_access is_trans (X,Y) @inbounds for i = 1:d
        z += X[x_pos,i] * Y[y_pos,i] * w[i]^2
    end
    z
end
=#

# Squared distance between vectors x and y
function sqdist{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    (n = length(x)) == length(y) || throw(ArgumentError("Dimensions do not conform."))
    z = zero(T)
    @inbounds @simd for i = 1:n
        v = x[i] - y[i]
        z += v*v
    end
    z
end
function sqdist{T<:FloatingPoint}(x::T, y::T)
    v = x - y
    v*v
end

#=
# In-place squared distance calculation
function sqdist!{T<:FloatingPoint}(d::Int64, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    z = zero(T)
    @transpose_access is_trans (X,Y) @inbounds for i = 1:d
        v = X[x_pos,i] - Y[y_pos,i]
        z += v*v
    end
    z
end
=#

# Weighted squared distance function between vectors x and y
function sqdist{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    z = zero(T)
    @inbounds @simd for i = 1:n
        v = (x[i] - y[i]) * w[i]
        z += v*v
    end
    z
end
function sqdist{T<:FloatingPoint}(x::T, y::T, w::T)
    v = w*(x - y)
    v*v
end

#=
# In-place weighted squared distance calculation
function sqdist!{T<:FloatingPoint}(d::Int64, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, w::Array{T}, is_trans::Bool)
    z = zero(T)
    @transpose_access is_trans (X,Y) @inbounds for i = 1:d
        v = (X[x_pos,i] - Y[y_pos,i]) * w[i]
        z += v*v
    end
    z
end
=#


#==========================================================================
  Gramian Functions
==========================================================================#

function init_gramian{T<:FloatingPoint}(X::Matrix{T}, is_trans::Bool = false)
    n = size(X, is_trans ? 2 : 1)
    Array(T, n, n)
end

function init_gramian{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
    dim = is_trans ? 2 : 1
    n = size(X, dim)
    m = size(Y, dim)
    Array(T, n, m)
end

safe_similar(X::Matrix) = similar(X)
safe_similar(X::Matrix{BigFloat}) = zeros(X)

## Calculate the scalar product matrix (matrix of scalar products)
#    is_trans == false -> Z = XXᵀ (X is a design matrix)
#                true  -> Z = XᵀX (X is a transposed design matrix)
function scprodmatrix!{T<:FloatingPoint}(Z::Matrix{T}, X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    (n = size(Z, 1)) == size(Z, 2) || throw(DimensionMismatch("Kernel matrix must be square."))
    if is_trans
        n == size(X, 2) || throw(DimensionMismatch("Supplied kernel matrix must be square with the same number of columns as X."))
        BLAS.syrk!(is_upper ? 'U' : 'L', 'T', one(T), X, zero(T), Z)  # 'T' -> C := αA'A + βC
    else
        n == size(X, 1) || throw(DimensionMismatch("Supplied kernel matrix must be square with the same number of rows as X."))
        BLAS.syrk!(is_upper ? 'U' : 'L', 'N', one(T), X, zero(T), Z)
    end
    sym ? (is_upper ? syml!(Z) : symu!(Z)) : Z
end
function scprodmatrix!(Z::Matrix{BigFloat}, X::Matrix{BigFloat}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    (n = size(Z, 1)) == size(Z, 2) || throw(DimensionMismatch("Kernel matrix must be square."))
    if is_trans
        n == size(X, 2) || throw(DimensionMismatch("Supplied kernel matrix must be square with the same number of columns as X."))
        m = size(X, 1)
    else
        n == size(X, 1) || throw(DimensionMismatch("Supplied kernel matrix must be square with the same number of rows as X."))
        m = size(X, 2)
    end
    @transpose_access is_trans (X,) @inbounds for j = 1:n
        for i = is_upper ? (1:j) : (j:n)
            v = zero(BigFloat)
            for k = 1:m
                v += X[i,k] * X[j,k]
            end
            Z[i,j] = v
        end
    end
    sym ? (is_upper ? syml!(Z) : symu!(Z)) : Z
end
function scprodmatrix{T<:FloatingPoint}(X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    scprodmatrix!(init_gramian(X, is_trans), X, is_trans, is_upper, sym)    
end

# Returns the upper right corner of the scalar product matrix of [Xᵀ Yᵀ]ᵀ or [X Y]
#   is_ trans == false -> Z = XYᵀ (X and Y are design matrices)
#                true  -> Z = XᵀY (X and Y are transposed design matrices)
function scprodmatrix!{T<:FloatingPoint}(Z::Matrix{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
    if is_trans
        size(X, 1) == size(Y, 1) || throw(DimensionMismatch("X must have as many rows as Y."))
        size(X, 2) == size(Z, 1) || throw(DimensionMismatch("Supplied kernel matrix must have as many rows as X has columns."))
        size(Y, 2) == size(Z, 2) || throw(DimensionMismatch("Supplied kernel matrix must have as many columns as Y has columns."))
        BLAS.gemm!('T', 'N', one(T), X, Y, zero(T), Z)
    else
        size(X, 2) == size(Y, 2) || throw(DimensionMismatch("X must have as many columns as Y."))
        size(X, 1) == size(Z, 1) || throw(DimensionMismatch("Supplied kernel matrix must have as many rows as X has rows."))
        size(Y, 1) == size(Z, 2) || throw(DimensionMismatch("Supplied kernel matrix must have as many columns as Y has rows."))
        BLAS.gemm!('N', 'T', one(T), X, Y, zero(T), Z)
    end
end
function scprodmatrix!(Z::Matrix{BigFloat}, X::Matrix{BigFloat}, Y::Matrix{BigFloat}, is_trans::Bool = false)
    if is_trans
        (m  = size(X, 1)) == size(Y, 1) || throw(DimensionMismatch("X must have as many rows as Y."))
        (nx = size(X, 2)) == size(Z, 1) || throw(DimensionMismatch("Supplied kernel matrix must have as many rows as X has columns."))
        (ny = size(Y, 2)) == size(Z, 2) || throw(DimensionMismatch("Supplied kernel matrix must have as many columns as Y has columns."))
    else
        (m  = size(X, 2)) == size(Y, 2) || throw(DimensionMismatch("X must have as many columns as Y."))
        (nx = size(X, 1)) == size(Z, 1) || throw(DimensionMismatch("Supplied kernel matrix must have as many rows as X has rows."))
        (ny = size(Y, 1)) == size(Z, 2) || throw(DimensionMismatch("Supplied kernel matrix must have as many columns as Y has rows."))
    end
    @transpose_access is_trans (X,Y) @inbounds for j = 1:ny
        for i = 1:nx
            v = zero(BigFloat)
            for k = 1:m
                v += X[i,k] * Y[j,k]
            end
            Z[i,j] = v
        end
    end
    Z
end
function scprodmatrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
    scprodmatrix!(init_gramian(X, Y, is_trans), X, Y, is_trans)
end

# Calculate the weighted scalar product matrix 
#    is_trans == false -> Z = XDXᵀ (X is a design matrix and D = diag(w))
#                true  -> Z = XᵀDX (X is a transposed design matrix and D = diag(w))
function scprodmatrix!{T<:FloatingPoint}(Z::Matrix{T}, X::Matrix{T}, w::Vector{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    scprodmatrix!(Z, is_trans ? scale(w, X) : scale(X, w), is_trans, is_upper, sym)
end
function scprodmatrix{T<:FloatingPoint}(X::Matrix{T}, w::Vector{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    scprodmatrix!(init_gramian(X, is_trans), X, w, is_trans, is_upper, sym)
end

# Returns the upper right corner of the scalar product matrix of [Xᵀ Yᵀ]ᵀ or [X Y]
#   is_trans == false -> Z = XDYᵀ (X and Y are design matrices)
#               true  -> Z = XᵀYD (X and Y are transposed design matrices)
function scprodmatrix!{T<:FloatingPoint}(Z::Matrix{T}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T}, is_trans::Bool = false)
    scprodmatrix!(Z, X, is_trans ? scale(w.^2, Y) : scale(Y, w.^2), is_trans)
end
function scprodmatrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, w::Vector{T}, is_trans::Bool = false)
    scprodmatrix!(init_gramian(X, Y, is_trans), X, Y, w, is_trans)
end
 
# Calculates Z such that Zij is the dot product of the difference of row i and j of matrix X
#    is_trans == false -> X is a design matrix
#                true  -> X is a transposed design matrix
function sqdistmatrix!{T<:FloatingPoint}(Z::Matrix{T}, X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    scprodmatrix!(Z, X, is_trans, is_upper, false)  # Don't symmetrize yet
    (n = size(Z, 1)) == size(Z, 2) || throw(DimensionMismatch("Z must be square."))
    xᵀx = diag(Z)
    length(xᵀx) == n || throw(DimensionMismatch(""))
    if is_upper
        @inbounds for j = 1:n, i = 1:j
            Z[i,j] = xᵀx[i] - 2Z[i,j] + xᵀx[j]
        end
        sym ? syml!(Z) : Z
    else
        @inbounds for j = 1:n, i = j:n
            Z[i,j] = xᵀx[i] - 2Z[i,j] + xᵀx[j]
        end
        sym ? symu!(Z) : Z
    end
end
function sqdistmatrix{T<:FloatingPoint}(X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    sqdistmatrix!(init_gramian(X, is_trans), X, is_trans, is_upper, sym)
end

# Calculates the upper right corner, Z, of the squared distance matrix of matrix [Xᵀ Yᵀ]ᵀ
#   is_trans == false -> X and Y are design matrices
#               true  -> X and Y are transposed design matrices
function sqdistmatrix!{T<:FloatingPoint}(Z::Matrix{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = true)
    scprodmatrix!(Z, X, Y, is_trans)
    n, m = size(Z)
    xᵀx = is_trans ? dot_columns(X) : dot_rows(X)
    n == length(xᵀx) || throw(DimensionMismatch(""))
    yᵀy = is_trans ? dot_columns(Y) : dot_rows(Y)
    m == length(yᵀy) || throw(DimensionMismatch(""))
    @inbounds for j = 1:m, i = 1:n
        Z[i,j] = xᵀx[i] - 2Z[i,j] + yᵀy[j]
    end
    Z
end
function sqdistmatrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
    sqdistmatrix!(init_gramian(X, Y, is_trans), X, Y, is_trans)
end

# Calculates Z such that Zij is the dot product of the difference of row i and j of matrix X
#    is_trans == false -> X is a design matrix
#                true  -> X is a transposed design matrix
function sqdistmatrix!{T<:FloatingPoint}(Z::Matrix{T}, X::Matrix{T}, w::Vector{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    scprodmatrix!(Z, X, w, is_trans, is_upper, false)
    (n = size(Z, 1)) == size(Z, 2) || throw(DimensionMismatch("Z must be square."))
    xᵀDx = diag(Z)
    n == length(xᵀDx) || throw(DimensionMismatch(""))
    if is_upper
        @inbounds for j = 1:n, i = 1:j
            Z[i,j] = xᵀDx[i] - 2Z[i,j] + xᵀDx[j]
        end
        sym ? syml!(Z) : Z
    else
        @inbounds for j = 1:n, i = j:n
            Z[i,j] = xᵀDx[i] - 2Z[i,j] + xᵀDx[j]
        end
        sym ? symu!(Z) : Z
    end
end
function sqdistmatrix{T<:FloatingPoint}(X::Matrix{T}, w::Vector{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    sqdistmatrix!(init_gramian(X, is_trans), X, w, is_trans, is_upper, sym)
end

# Calculates the upper right corner, Z, of the squared distance matrix of matrix [Xᵀ Yᵀ]ᵀ
#   is_trans == false -> X and Y are design matrices
#               true  -> X and Y are transposed design matrices
function sqdistmatrix!{T<:FloatingPoint}(Z::Matrix{T}, X::Matrix{T}, Y::Matrix{T}, w::Vector{T}, is_trans::Bool = false)
    w² = vec(w.^2)
    scprodmatrix!(Z, X, is_trans ? scale(w², Y) : scale(Y, w²), is_trans)
    n, m = size(Z)
    xᵀDx = is_trans ? dot_columns(X, w²) : dot_rows(X, w²)
    n == length(xᵀDx) || throw(DimensionMismatch(""))
    yᵀDy = is_trans ? dot_columns(Y, w²) : dot_rows(Y, w²)
    m == length(yᵀDy) || throw(DimensionMismatch(""))
    @inbounds for j = 1:m, i = 1:n
        Z[i,j] = xᵀDx[i] - 2Z[i,j] + yᵀDy[j]
    end
    Z
end
function sqdistmatrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, w::Vector{T}, is_trans::Bool = false)
    sqdistmatrix!(init_gramian(X, Y, is_trans), X, Y, w, is_trans)
end
