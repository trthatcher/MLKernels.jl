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
        @inbounds for j = 1:(p - 1) 
            for i = (j + 1):p 
                S[i, j] = S[j, i]
            end
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
        @inbounds for j = 2:p
            for i = 1:j-1
                S[i,j] = S[j,i]
            end 
        end
    end
    return S
end
symu(S::Matrix) = symu!(copy(S))

# Return vector of dot products for each row of A
function dot_rows{T<:FloatingPoint}(A::Matrix{T})
    n, m = size(A)
    aᵀa = zeros(T, n)
    @inbounds for j = 1:m
        for i = 1:n
            aᵀa[i] += A[i,j]*A[i,j]
        end
    end
    aᵀa
end

# Return vector of dot products for each row of A
function dot_columns{T<:FloatingPoint}(A::Matrix{T})
    n, m = size(A)
    aᵀa = zeros(T, m)
    @inbounds for j = 1:m
        for i = 1:n
            aᵀa[j] += A[i,j]*A[i,j]
        end
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

# Diagonal-General Matrix Multiply: overwrites A with DA (scales A[i,:] by D[i,i])
function dgmm!{T<:FloatingPoint}(D::Array{T}, A::Matrix{T})
    n, p = size(A)
    if n != (d = length(D))
        throw(ArgumentError(
            "The length(D) = $(d) of vector D representing a $(d)×$(d) diagonal matrix must " * (
            "equal the number of rows in Matrix A ∈ ℝ$(n)×$(p) to compute diagonal matrix " * (
            "product B = DA"))
        ))
    end
    @inbounds for j = 1:p
        for i = 1:n
            A[i, j] *= D[i]
        end
    end
    A
end
dgmm(D,A) = dgmm!(D, copy(A))

# General-Diagonal Matrix Multiply: overwrites A with AD (scales A[:,i] by D[i,i]
function gdmm!{T<:FloatingPoint}(A::Matrix{T}, D::Array{T})
    n, p = size(A)
    if p != (d = length(D))
        throw(ArgumentError(
            "The length(D) = $(d) of vector D representing a $(d)×$(d) diagonal matrix must " * (
            "equal the number of columns in Matrix A ∈ ℝ$(n)×$(p) to compute diagonal matrix " * (
            "product B = AD"))
        ))
    end
    @inbounds for j = 1:p
        for i = 1:n
            A[i, j] *= D[j]
        end
    end
    A
end
gdmm(A,D) = gdmm!(copy(A), D)

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
    0 <= ϵ || throw(ArgumentError("ϵ = $(ϵ) must be in [0,∞)"))
    @inbounds for i = 1:p
        S[i,i] += ϵ
    end
    S
end
perturb{T<:FloatingPoint}(S::Matrix{T}, ϵ::T) = perturb!(copy(S),ϵ)
perturb{T<:FloatingPoint}(S::Matrix{T}) = perturb!(copy(S))


#==========================================================================
  Array Computations
==========================================================================#

#### Helper Functions for coordinate epsilons
####     In-place calculation of ϵ = x - y for specified X and Y coordinates
    function N_epsilon!{T<:FloatingPoint}(E::Array{T}, X::Matrix{T}, Y::Matrix{T}, d::Integer, x_pos::Integer, y_pos::Integer)
        @inbounds for i = 1:d
            E[i, x_pos, y_pos] = X[x_pos,i] - Y[y_pos,i]
        end
        E
    end
    function T_epsilon!{T<:FloatingPoint}(E::Array{T}, X::Matrix{T}, Y::Matrix{T}, d::Integer, x_pos::Integer, y_pos::Integer)
        @inbounds for i = 1:d
            E[i, x_pos, y_pos] = X[i,x_pos] - Y[i,y_pos]
        end
        E
    end
####

# epsilons: Calculates the set of ϵ vectors for every pair of vectors in X and Y
#     [:,i,j] = ϵ = { X[i,:] - Y[j,:]   if trans == 'N'
#                   { X[:,i] - Y[:,j]   if trans == 'T'
function epsilons{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    is_trans = trans == 'T'  # True if columns are observations
    n = size(X, is_trans ? 2 : 1)
    m = size(Y, is_trans ? 2 : 1)
    if (d = size(X, is_trans ? 1 : 2)) != size(Y, is_trans ? 1 : 2)
        throw(ArgumentError("X and Y do not have the same number of " * is_trans ? "rows." : "columns."))
    end
    E = Array(T, d, n, m)
    if is_trans
        @inbounds for j = 1:m
            for i = 1:n
                T_epsilon!(E, X, Y, d, i, j)
            end
        end
    else
        @inbounds for j = 1:m
            for i = 1:n
                N_epsilon!(E, X, Y, d, i, j)
            end
        end
    end
    E
end

#### Helper Functions for difference_elements
####     In-place calculation of the tensor product of each epsilon with itself
    function T_tensor_epsilon!{T<:FloatingPoint}(E::Array{T}, X::Matrix{T}, Y::Matrix{T}, d::Integer, x_pos::Integer, y_pos::Integer)
        @inbounds for j = 1:d
            ϵ = X[j,x_pos] - Y[j,y_pos]
            for i = 1:d
                E[i,j,x_pos,y_pos] = (X[i,x_pos] - Y[i,y_pos]) * ϵ
            end
        end
        E
    end
    function N_tensor_epsilon!{T<:FloatingPoint}(E::Array{T}, X::Matrix{T}, Y::Matrix{T}, d::Integer, x_pos::Integer, y_pos::Integer)
        @inbounds for j = 1:d
            ϵ = X[x_pos,j] - Y[y_pos,j]
            for i = 1:d
                E[i,j,x_pos,y_pos] = (X[x_pos,i] - Y[y_pos,i]) * ϵ
            end
        end
        E
    end
####

# epsilons: Calculates the set of ϵϵᵀ (tensor/outer product) matrices for every pair of vectors in X and Y
#     [:,:,i,j] = ϵϵᵀ where ϵ = { X[i,:] - Y[j,:]   if trans == 'N'
#                               { X[:,i] - Y[:,j]   if trans == 'T'
function tensor_epsilons{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    is_trans = trans == 'T'  # True if columns are observations
    n = size(X, is_trans ? 2 : 1)
    m = size(Y, is_trans ? 2 : 1)
    if (d = size(X, is_trans ? 1 : 2)) != size(Y, is_trans ? 1 : 2)
        throw(ArgumentError("X and Y do not have the same number of " * (is_trans ? "rows." : "columns.")))
    end
    E = Array(T, d, d, n, m)
    if is_trans
        @inbounds for j = 1:m
            for i = 1:n
                T_tensor_epsilon!(E, X, Y, d, i, j)
            end
        end
    else
        @inbounds for j = 1:m
            for i = 1:n
                N_tensor_epsilon!(E, X, Y, d, i, j)
            end
        end
    end
    E
end
