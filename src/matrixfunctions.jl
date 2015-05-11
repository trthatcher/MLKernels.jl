#===================================================================================================
  Matrix Functions
===================================================================================================#

#==========================================================================
  Auxiliary Functions
==========================================================================#

# Symmetrize the lower half of matrix S using the upper half of S
function syml!(S::Matrix)
    p = size(S, 1)
    p == size(S, 2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
    if p > 1 
        @inbounds for j = 1:(p - 1), i = (j + 1):p 
                S[i, j] = S[j, i]
        end
    end
    return S
end
syml(S::Matrix) = syml!(copy(S))

# Symmetrize the upper off-diagonal of matrix S using the lower half of S
function symu!(S::Matrix)
    p = size(S,1)
    p == size(S,2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
    if p > 1 
        @inbounds for j = 2:p, i = 1:j-1
                S[i,j] = S[j,i]
        end
    end
    return S
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

# Add array z to each row in X, overwrites and returns X
function row_add!{T<:FloatingPoint}(X::Matrix{T}, z::Array{T})
    n, p = size(X)
    p == length(z) || throw(ArgumentError("Dimensions do not conform"))
    @inbounds for j = 1:p
        for i = 1:n
            X[i,j] += z[j]
        end
    end
    X
end

# Add array z to each column in X, overwrites and returns X
function col_add!{T<:FloatingPoint}(X::Matrix{T}, z::Array{T})
    n, p = size(X)
    p == length(z) || throw(ArgumentError("Dimensions do not conform"))
    @inbounds for j = 1:p
        for i = 1:n
            X[i,j] += z[i]
        end
    end
    X
end

# Subtract array z from each row in X, overwrites and returns X
function row_sub!{T<:FloatingPoint}(X::Matrix{T}, z::Array{T})
    n, p = size(X)
    p == length(z) || throw(ArgumentError("Dimensions do not conform"))
    @inbounds for j = 1:p
        for i = 1:n
            X[i,j] -= z[j]
        end
    end
    X
end

# Subtract array z from each column in X, overwrites and returns X
function col_sub!{T<:FloatingPoint}(X::Matrix{T}, z::Array{T})
    n, p = size(X)
    p == length(z) || throw(ArgumentError("Dimensions do not conform"))
    @inbounds for j = 1:p
        for i = 1:n
            X[i,j] -= z[i]
        end
    end
    X
end


#==========================================================================
  Matrix Operations
==========================================================================#

# Overwrite A with the hadamard product of A and B. Returns A
function hadamard!{T<:FloatingPoint}(A::Array{T}, B::Array{T})
    length(A) == length(B) || error("A and B must be of the same length.")
    @inbounds for i = 1:length(A)
        A[i] *= B[i]
    end
    A
end

# Overwrite A with the hadamard product of A and B. Returns A
function hadamard!{T<:FloatingPoint}(A::Matrix{T}, B::Matrix{T}, uplo::Char, sym::Bool = true)
    n = size(A,1)
    if !(n == size(A,2) == size(B,1) == size(B,2))
        throw(ArgumentError("A and B must be square and of same order."))
    end
    @inbounds for j = 1:n
        for i = uplo == 'U' ? (1:j) : (j:n)
            A[i,j] *= B[i,j]
        end 
    end
    sym ? (uplo == 'U' ? syml!(A) : symu!(A)) : A
end

# Regularize a square matrix S, overwrites S with (1-α)*S + α*β*I
function regularize!{T<:FloatingPoint}(S::Matrix{T}, α::T, β::T=trace(S)/size(S,1))
    (p = size(S,1)) == size(S,2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
    0 <= α <= 1 || throw(ArgumentError("α=$(α) must be in the interval [0,1]"))
    @inbounds for j = 1:p
        for i = 1:p
            S[i,j] *= one(T) - α
        end
        S[j,j] += α*β
    end
    S
end
regularize{T<:FloatingPoint}(S::Matrix{T}, α::T) = regularize!(copy(S), α)
regularize{T<:FloatingPoint}(S::Matrix{T}, α::T, β::T) = regularize!(copy(S), α, β)

# Perturb a symmetric matrix S, overwrites S with S + ϵ*I
function perturb!{T<:FloatingPoint}(S::Matrix{T}, ϵ::T = 100*eps(T)*size(S,1))
    (p = size(S,1)) == size(S,2) || throw(ArgumentError("S ∈ ℝ$(p)×$(size(S, 2)) must be square"))
    #0 <= ϵ || throw(ArgumentError("ϵ = $(ϵ) must be in [0,∞)"))
    @inbounds for i = 1:p
        S[i,i] += ϵ
    end
    S
end
perturb{T<:FloatingPoint}(S::Matrix{T}, ϵ::T) = perturb!(copy(S),ϵ)
perturb{T<:FloatingPoint}(S::Matrix{T}) = perturb!(copy(S))
