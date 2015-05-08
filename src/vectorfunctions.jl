#===================================================================================================
  Vector Operations
===================================================================================================#

#==========================================================================
  Scalar Product Function (unweighted)
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

# Partial derivative of the scalar product of vectors x and y
scprod_dx{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = copy(y)
scprod_dy{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = copy(x)

# Calculate the scalar product matrix (matrix of scalar products)
#    trans == 'N' -> Z = XXᵀ (X is a design matrix)
#             'T' -> Z = XᵀX (X is a transposed design matrix)
function scprodmatrix{T<:FloatingPoint}(X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
    Z = BLAS.syrk(uplo, trans, one(T), X)
    sym ? (uplo == 'U' ? syml!(Z) : symu!(Z)) : Z
end

# Returns the upper right corner of the scalar product matrix of [Xᵀ Yᵀ]ᵀ or [X Y]
#   trans == 'N' -> Z = XYᵀ (X and Y are design matrices)
#            'T' -> Z = XᵀY (X and Y are transposed design matrices)
function scprodmatrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    BLAS.gemm(trans, trans == 'N' ? 'T' : 'N', X, Y)
end


#==========================================================================
  Scalar Product Function (weighted)
==========================================================================#

# Weighted scalar product of x and y
function scprod{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    z = zero(T)
    @inbounds @simd for i = 1:n
        z += x[i]*y[i]*w[i]
    end
    z
end

function scprod_dx!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    @inbounds @simd for i = 1:n
        x[i] = y[i]*w[i]
    end
    x
end

scprod_dy!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = scprod_dx!(y, x, w)
scprod_dw!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = scprod_dx!(w, x, y)

scprod_dx{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = scprod_dx!(similar(x), y, w)
scprod_dy{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = scprod_dy!(x, similar(y), w)
scprod_dw{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = scprod_dw!(x, y, similar(w))

# Calculate the gramian 
#    trans == 'N' -> Z = XDXᵀ (X is a design matrix and D = diag(w))
#             'T' -> Z = XᵀDX (X is a transposed design matrix and D = diag(w))
function scprodmatrix{T<:FloatingPoint}(X::Matrix{T}, w::Array{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
    Z = BLAS.syrk(uplo, trans, one(T), trans == 'T' ? scale(vec(sqrt(w)), X) : scale(X, vec(sqrt(w))))
    sym ? (uplo == 'U' ? syml!(Z) : symu!(Z)) : Z
end

# Returns the upper right corner of the scalar product matrix of [Xᵀ Yᵀ]ᵀ or [X Y]
#   trans == 'N' -> G = XDYᵀ (X and Y are design matrices)
#            'T' -> G = XᵀDY (X and Y are transposed design matrices)
function scprodmatrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, w::Array{T}, trans::Char = 'N')
    BLAS.gemm(trans, trans == 'N' ? 'T' : 'N', X, trans == 'T' ? scale(vec(w), Y) : scale(Y, vec(w)))
end


#==========================================================================
  Squared Distance Function (unweighted)
==========================================================================#
 
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

sqdist_dx{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = scale!(2, x - y)
sqdist_dy{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = scale!(2, y - x)

# Calculates G such that Gij is the dot product of the difference of row i and j of matrix X
#    trans == 'N' -> X is a design matrix
#             'T' -> X is a transposed design matrix
function sqdistmatrix{T<:FloatingPoint}(X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
    Z = scprodmatrix(X, trans, uplo, false)
    n = size(X, trans == 'N' ? 1 : 2)
    xᵀx = copy(vec(diag(Z)))
    @inbounds for j = 1:n
        for i = uplo == 'U' ? (1:j) : (j:n)
            Z[i,j] = xᵀx[i] - 2Z[i,j] + xᵀx[j]
        end
    end
    sym ? (uplo == 'U' ? syml!(Z) : symu!(Z)) : Z
end

# Calculates the upper right corner G of the squared distance matrix of matrix [Xᵀ Yᵀ]ᵀ
#   trans == 'N' -> X and Y are design matrices
#            'T' -> X and Y are transposed design matrices
function sqdistmatrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    n = size(X, trans == 'N' ? 1 : 2)
    m = size(Y, trans == 'N' ? 1 : 2)
    xᵀx = trans == 'N' ? dot_rows(X) : dot_columns(X)
    yᵀy = trans == 'N' ? dot_rows(Y) : dot_columns(Y)
    Z = scprodmatrix(X, Y, trans)
    @inbounds for j = 1:m
        for i = 1:n
            Z[i,j] = xᵀx[i] - 2Z[i,j] + yᵀy[j]
        end
    end
    Z
end


#==========================================================================
  Squared Distance Function (weighted)
==========================================================================#

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

function sqdist_dx!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    @inbounds @simd for i = 1:n
        x[i] = 2(x[i] - y[i]) * w[i]^2
    end
    x
end

sqdist_dy!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = sqdist_dx!(y, x, w)

function sqdist_dw!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    @inbounds @simd for i = 1:n
        w[i] = 2(x[i] - y[i])^2 * w[i]
    end
    w
end

sqdist_dx{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = sqdist_dx!(copy(x), y, w)
sqdist_dy{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = sqdist_dy!(x, copy(y), w)
sqdist_dw{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = sqdist_dw!(x, y, copy(w))

# Calculates G such that Gij is the dot product of the difference of row i and j of matrix X
#    trans == 'N' -> X is a design matrix
#             'T' -> X is a transposed design matrix
function sqdistmatrix{T<:FloatingPoint}(X::Matrix{T}, w::Array{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
    Z = BLAS.syrk(uplo, trans, one(T), trans == 'T' ? scale(vec(w), X) : scale(X, vec(w)))
    n = size(X, trans == 'N' ? 1 : 2)
    xᵀDx = copy(vec(diag(Z)))
    @inbounds for j = 1:n
        for i = uplo == 'U' ? (1:j) : (j:n)
            Z[i,j] = xᵀDx[i] - 2Z[i,j] + xᵀDx[j]
        end
    end
    sym ? (uplo == 'U' ? syml!(Z) : symu!(Z)) : Z
end

# Calculates the upper right corner G of the squared distance matrix of matrix [Xᵀ Yᵀ]ᵀ
#   trans == 'N' -> X and Y are design matrices
#            'T' -> X and Y are transposed design matrices
function sqdistmatrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, w::Array{T}, trans::Char = 'N')
    n = size(X, trans == 'N' ? 1 : 2)
    m = size(Y, trans == 'N' ? 1 : 2)
    w² = vec(w.^2)
    xᵀDx = trans == 'N' ? dot_rows(X, w²) : dot_columns(X, w²)
    yᵀDy = trans == 'N' ? dot_rows(Y, w²) : dot_columns(Y, w²)
    Z = BLAS.gemm(trans, trans == 'N' ? 'T' : 'N', X, trans == 'T' ? scale(w², Y) : scale(Y, w²))
    @inbounds for j = 1:m
        for i = 1:n
            Z[i,j] = xᵀDx[i] - 2Z[i,j] + yᵀDy[j]
        end
    end
    Z
end
