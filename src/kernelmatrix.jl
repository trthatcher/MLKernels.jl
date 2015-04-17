#===================================================================================================
  Kernel Matrices
===================================================================================================#

#==========================================================================
  Auxiliary Functions
==========================================================================#

# Symmetrize the lower half of matrix S using the upper half of S
function syml!(S::Matrix)
    p = size(S, 1)
    p == size(S, 2) || error("S ∈ ℝ$(p)×$(size(S, 2)) should be square")
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

# Overwrite A with the hadamard product of A and B. Returns A
function hadamard!{T<:FloatingPoint}(A::Array{T}, B::Array{T})
    length(A) == length(B) || error("Dimensions do not conform.")
    @inbounds for i = 1:length(A)
        A[i] *= B[i]
    end
    A
end


#==========================================================================
  Gramian Functions
==========================================================================#

# Calculate the gramian
#    If trans = 'N' then G = XXᵀ (X is a design matrix)
#    If trans = 'T' then G = XᵀX (X is a transposed design matrix)
function gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U',
                                          sym::Bool = true)
    G = BLAS.syrk(uplo, trans, one(T), X)
    sym ? (uplo == 'U' ? syml!(G) : symu!(G)) : G
end

# Calculate the upper right corner of the gramian matrix of [Xᵀ Yᵀ]ᵀ
#   If trans = 'N' then G = XYᵀ (X and Y are design matrices)
#   If trans = 'T' then G = XYᵀ (X and Y are transposed design matrices)
function gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    G::Array{T} = BLAS.gemm(trans, trans == 'N' ? 'T' : 'N', X, Y)
end

# Calculates G such that Gij is the dot product of the difference of row i and j of matrix X
function lagged_gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U',
                                                 sym::Bool = true)
    G = gramian_matrix(X, trans, uplo, false)
    n = size(X, trans == 'N' ? 1 : 2)
    xᵀx = copy(vec(diag(G)))
    for j = 1:n
        for i = uplo == 'U' ? (1:j) : (j:n)
            println(i, ",", j)
            G[i,j] = xᵀx[i] - convert(T, 2) * G[i,j] + xᵀx[j]
        end
    end
    sym ? (uplo == 'U' ? syml!(G) : symu!(G)) : G
end

# Calculates the upper right corner G of the lagged gramian matrix of matrix [Xᵀ Yᵀ]ᵀ
function lagged_gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    n = size(X, trans == 'N' ? 1 : 2)
    m = size(Y, trans == 'N' ? 1 : 2)
    xᵀx = trans == 'N' ? dot_rows(X) : dot_columns(X)
    yᵀy = trans == 'N' ? dot_rows(Y) : dot_columns(Y)
    G = gramian_matrix(X, Y, trans)
    @inbounds for j = 1:m
        for i = 1:n
            G[i,j] = xᵀx[i] - convert(T, 2) * G[i,j] + yᵀy[j]
        end
    end
    G
end

# Centralize a kernel matrix K
function center_kernel_matrix!{T<:FloatingPoint}(K::Matrix{T})
	n = size(K, 1)
	n == size(K, 2) || error("Kernel matrix must be square")
	row_mean = sum(K, 1)
	element_mean = sum(row_mean) / (convert(T, n)^2)
	BLAS.scal!(n, one(T)/convert(T,n), row_mean, 1)
	((K .- row_mean) .- row_mean') .+ element_mean
end
center_kernel_matrix{T<:FloatingPoint}(K::Matrix{T}) = center_kernel_matrix!(copy(K))


#==========================================================================
  Generic Kernel Matrix Functions
==========================================================================#

# Generic kernel matrix function - will be slow
function kernel_matrix{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T})
    n = size(X, 1)
    K::Matrix{T} = Array(T, n, n)
    @inbounds for i = 1:n 
        for j = i:n
            K[i,j] = kernel_function(κ, vec(X[i,:]), vec(X[j,:]))::T
        end 
    end
    syml!(K)
end

function kernel_matrix_scaled{T<:FloatingPoint}(a::T, κ::StandardKernel{T}, X::Matrix{T})
    K::Matrix{T} = kernel_matrix(κ, X)
    if a != one(T) BLAS.scal!(length(K), a, K, 1) end
    K
end

function kernel_matrix_product{T<:FloatingPoint}(a::T, κ₁::StandardKernel{T},
                                                 κ₂::StandardKernel{T}, X::Matrix{T})
    K::Matrix{T} = kernel_matrix_scaled(a, κ₁, X)
    hadamard!(K, kernel_matrix(κ₂, X))
end

function kernel_matrix_sum{T<:FloatingPoint}(a₁::T, κ₁::StandardKernel{T}, a₂::T, 
                                             κ₂::StandardKernel{T}, X::Matrix{T})
    K::Matrix{T} = kernel_matrix_scaled(a₁, κ₁, X)
    BLAS.axpy!(length(K), a₂, kernel_matrix(κ₂, X), 1, K, 1)    
end

function kernel_matrix{T<:FloatingPoint}(ψ::ScaledKernel{T}, X::Matrix{T})
    kernel_matrix_scaled(ψ.a, ψ.κ, X)
end
function kernel_matrix{T<:FloatingPoint}(ψ::KernelProduct{T}, X::Matrix{T})
    kernel_matrix_product(ψ.a, ψ.κ₁, ψ.κ₂, X)
end
function kernel_matrix{T<:FloatingPoint}(ψ::KernelSum{T}, X::Matrix{T})
    kernel_matrix_sum(ψ.a₁, ψ.κ₁, ψ.a₂, ψ.κ₂, X)
end

function kernel_matrix{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T})
    n = size(X, 1)
    m = size(Y, 1)
    size(X, 2) == size(Z, 2) || error("X ∈ ℝn×p and Y should be ∈ ℝm×p, but X ∈ " * (
                                      "ℝn×$(size(Y, 2)) and Y∈ ℝm×$(size(Y, 2))."))
    K::Matrix{T} = Array(T, n, m)
    @inbounds for j = 1:m 
        for i = 1:n
            K[i,j] = kernel_function(κ, vec(X[i,:]), vec(Y[j,:]))::T
        end
    end
    K
end

function kernel_matrix_scaled{T<:FloatingPoint}(a::T, κ::StandardKernel{T}, X::Matrix{T}, 
                                                Y::Matrix{T})
    K::Matrix{T} = kernel_matrix(κ, X, Y)
    if a != one(T) BLAS.scal!(length(K), a, K, 1) end
    K
end

function kernel_matrix_product{T<:FloatingPoint}(a::T, κ₁::StandardKernel{T},
                                                 κ₂::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T})
    K::Matrix{T} = kernel_matrix_scaled(a, κ₁, X, Y)
    hadamard!(K, kernel_matrix(κ₂, X, Y))
end

function kernel_matrix_sum{T<:FloatingPoint}(a₁::T, κ₁::StandardKernel{T}, a₂::T, 
                                             κ₂::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T})
    K::Matrix{T} = kernel_matrix_scaled(a₁, κ₁, X, Y)
    axpy!(length(K), a₂, kernel_matrix(κ₂, X, Y), 1, K, 1)    
end

function kernel_matrix{T<:FloatingPoint}(ψ::ScaledKernel{T}, X::Matrix{T}, Y::Matrix{T})
    kernel_matrix_scaled(ψ.a, ψ.κ, X, Y)
end
function kernel_matrix{T<:FloatingPoint}(ψ::KernelProduct{T}, X::Matrix{T}, Y::Matrix{T})
    kernel_matrix_product(ψ.a, ψ.κ₁, ψ.κ₂, X, Y)
end
function kernel_matrix{T<:FloatingPoint}(ψ::KernelSum{T}, X::Matrix{T}, Y::Matrix{T})
    kernel_matrix_sum(ψ.a₁, ψ.κ₁, ψ.a₂, ψ.κ₂, X, Y)
end


#==========================================================================
  Optimized kernel matrix functions for Euclidean distance and scalar
  product kernels
==========================================================================#


for (kernel, gramian) in ((:EuclideanDistanceKernel, :lagged_gramian_matrix),
                          (:ScalarProductKernel, :gramian_matrix))
    @eval begin
        function kernelize_gramian!{T<:FloatingPoint}(G::Array{T}, κ::$kernel{T})
            @inbounds for i = 1:length(G)
                G[i] = kernelize_scalar(κ, G[i])
            end
            G
        end

        function kernel_matrix{T<:FloatingPoint}(κ::$kernel{T}, X::Matrix{T})
            G::Matrix{T} = $gramian(X)
            K::Matrix{T} = kernelize_gramian!(G, κ)
        end

        function kernel_matrix_scaled{T<:FloatingPoint}(a::T, κ::$kernel{T}, X::Matrix{T})
            G::Matrix{T} = $gramian(X)
            K::Matrix{T} = kernelize_gramian!(G, κ)
            if a != one(T) BLAS.scal!(length(K), a, K, 1) end
            K
        end

        function kernel_matrix_product{T<:FloatingPoint}(a::T, κ₁::$Kernel{T}, κ₂::$Kernel{T},
                                                         X::Matrix{T})
            G::Matrix{T} = $gramian(X)
            K::Matrix{T} = kernelize_gramian!(copy(G), κ₁)
            if a != one(T) BLAS.scal!(length(K), a, K, 1) end
            hadamard!(K, kernelize_gramian!(G, κ₂))
        end

        function kernel_matrix_sum{T<:FloatingPoint}(a₁::T, κ₁::$Kernel{T}, a₂::T, κ₂::$Kernel{T},
                                                     X::Matrix{T})
            G::Matrix{T} = $gramian(X)
            K::Matrix{T} = kernelize_gramian!(copy(G), κ₁)
            n = length(K)
            if a₁ != one(T) BLAS.scal!(n, a₁, K, 1) end
            BLAS.axpy!(n, a₂, kernelize_gramian!(G, κ₂), 1, K, 1)
        end

        function kernel_matrix{T<:FloatingPoint}(κ::$kernel{T}, X::Matrix{T}, Y::Matrix{T})
            G::Matrix{T} = $gramian(X, Y)
            K::Matrix{T} = kernelize_gramian!(G, κ)
        end

        function kernel_matrix_scaled{T<:FloatingPoint}(a::T, κ::$kernel{T}, X::Matrix{T},
                                                        Y::Matrix{T})
            G::Matrix{T} = $gramian(X, Y)
            K::Matrix{T} = kernelize_gramian!(G, κ)
            if a != one(T) BLAS.scal!(length(K), a, K, 1) end
            K
        end

        function kernel_matrix_product{T<:FloatingPoint}(a::T, κ₁::$Kernel{T}, κ₂::$Kernel{T},
                                                         X::Matrix{T}, Y::Matrix{T})
            G::Matrix{T} = $gramian(X, Y)
            K::Matrix{T} = kernelize_gramian!(copy(G), κ₁)
            if a != one(T) BLAS.scal!(length(K), a, K, 1) end
            hadamard!(K, kernelize_gramian!(G, κ₂))
        end

        function kernel_matrix_sum{T<:FloatingPoint}(a₁::T, κ₁::$Kernel{T}, a₂::T, κ₂::$Kernel{T},
                                                     X::Matrix{T}, Y::Matrix{T})
            G::Matrix{T} = $gramian(X, Y)
            K::Matrix{T} = kernelize_gramian!(copy(G), κ₁)
            n = length(K)
            if a₁ != one(T) BLAS.scal!(n, a₁, K, 1) end
            BLAS.axpy!(n, a₂, kernelize_gramian!(G, κ₂), 1, K, 1)
        end
    end
end


#==========================================================================
  Optimized kernel matrix functions for Separable kernels
==========================================================================#

for kernel in (:MercerSigmoidKernel,)
    @eval begin

        function kernel_matrix_scaled{T<:FloatingPoint}(a::T, κ::$kernel{T}, X::Matrix{T})
            K::Matrix{T} = BLAS.syrk('U', 'N', a, kernelize_vector!(κ, copy(X)))
            syml!(K)
        end

        function kernel_matrix{T<:FloatingPoint}(κ::$kernel{T}, X::Matrix{T})
            kernel_matrix_scaled!(one(T), κ, X)
        end

        function kernel_matrix_scaled{T<:FloatingPoint}(a::T, κ::$kernel{T}, X::Matrix{T},
                                                        Y::Matrix{T})
            K::Array{T} = BLAS.gemm('N', 'T', a, kernelize_vector!(κ, copy(X)), 
                                                 kernelize_vector!(κ, copy(Y)))
        end

        function kernel_matrix{T<:FloatingPoint}(κ::$kernel{T}, X::Matrix{T}, Y::Matrix{T})
            kernel_matrix_scaled!(one(T), κ, X, Y)
        end

    end
end


#===================================================================================================
  Kernel Approximation
===================================================================================================#

# Diagonal-General Matrix Multiply 
function dgmm!{T<:FloatingPoint}(D::Array{T}, A::Matrix{T})
    n, p = size(A)
    n == length(D) || error("Diagonal matrix is not of correct dimension")
    @inbounds for j = 1:p
        for i = 1:n
            A[i, j] *= D[i]
        end
    end
    A
end

# Moore-Penrose pseudo-inverse for positive semidefinite matrices
function pinv_semiposdef!{T<:FloatingPoint}(S::Matrix{T}, tol::T = eps(T)*maximum(size(S)))
    tol > 0 || error("tol = $tol must be a positive number.")
    n = size(S,1)
    n == size(S,2) || error("not cool")
    UDVᵀ = svdfact!(S, thin=true)
    Σ⁻¹::Vector{T} = UDVᵀ[:S]
	@inbounds for i = 1:n
		Σ⁻¹[i] = Σ⁻¹[i] < tol ? zero(T) : one(T) / sqrt(Σ⁻¹[i])
	end
    Vᵀ::Matrix{T} = UDVᵀ[:Vt]
    dgmm!(Σ⁻¹, Vᵀ)::Matrix{T}
end

# Nystrom method for Kernel Matrix approximation
function nystrom{T<:FloatingPoint,S<:Integer}(κ::Kernel{T}, X::Matrix{T}, sₓ::Array{S})
    c = length(sₓ)
    n = size(X, 1)
    C::Matrix{T} = kernel_matrix(κ, X, X[sₓ,:])
    P::Matrix{T} = pinv_semiposdef!(C[sₓ,:])  # P'P = W = pinv(X[sₓ,sₓ]) 
    PCᵀ::Matrix{T} = BLAS.gemm('N', 'T', P, C)
    K = BLAS.syrk('U', 'T', one(T), PCᵀ)
    syml!(K)::Matrix{T}
end
