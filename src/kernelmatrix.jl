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

# Overwrite A with the hadamard product of A and B. Returns A
function hadamard!{T<:FloatingPoint}(A::Array{T}, B::Array{T}, uplo::Char, sym::Bool = true)
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

# Calculate the upper right corner of the gramian matrix of [Xᵀ Yᵀ]ᵀ or [X Y]
#   If trans = 'N' then G = XYᵀ (X and Y are design matrices)
#   If trans = 'T' then G = XᵀY (X and Y are transposed design matrices)
function gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    G::Array{T} = BLAS.gemm(trans, trans == 'N' ? 'T' : 'N', X, Y)
end

# Calculates G such that Gij is the dot product of the difference of row i and j of matrix X
function lagged_gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U',
                                                 sym::Bool = true)
    G = gramian_matrix(X, trans, uplo, false)
    n = size(X, trans == 'N' ? 1 : 2)
    xᵀx = copy(vec(diag(G)))
    @inbounds for j = 1:n
        for i = uplo == 'U' ? (1:j) : (j:n)
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
function center_kernelmatrix!{T<:FloatingPoint}(K::Matrix{T})
	n = size(K, 1)
	n == size(K, 2) || error("Kernel matrix must be square")
	row_mean = sum(K, 1)
	element_mean = sum(row_mean) / (convert(T, n)^2)
	BLAS.scal!(n, one(T)/convert(T,n), row_mean, 1)
	((K .- row_mean) .- row_mean') .+ element_mean
end
center_kernelmatrix{T<:FloatingPoint}(K::Matrix{T}) = center_kernelmatrix!(copy(K))


#==========================================================================
  Generic Kernel Matrix Functions
==========================================================================#

# Returns kernel matrix using generic approach - will be slow
function kernelmatrix{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, trans::Char = 'N',
                                        uplo::Char = 'U', sym::Bool = true)
    n = size(X, trans == 'N' ? 1 : 2)
    K = Array(T, n, n)
    if trans == 'N'
        @inbounds for j = 1:n
            for i = uplo == 'U' ? (1:j) : (j:n)
                K[i,j] = kernel(κ, X[i,:], X[j,:])
            end 
        end
    else
        @inbounds for j = 1:n
            for i = uplo == 'U' ? (1:j) : (j:n)
                K[i,j] = kernel(κ, X[:,i], X[:,j])
            end 
        end
    end
    sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K
end

# Returns kernel matrix scaled by a
function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::StandardKernel{T}, X::Matrix{T},
                                               trans::Char = 'N', uplo::Char = 'U', 
                                               sym::Bool = true)
    K = kernelmatrix(κ, X, trans, uplo, sym)
    a == one(T) ? K : BLAS.scal!(length(K), a, K, 1)
end

# Returns the kernel matrix of X for a product of two kernels
function kernelmatrix_product{T<:FloatingPoint}(a::T, κ₁::StandardKernel{T},
                                                κ₂::StandardKernel{T}, X::Matrix{T}, 
                                                trans::Char = 'N', uplo::Char = 'U', 
                                                sym::Bool = true)
    K = kernelmatrix_scaled(a, κ₁, X, trans, uplo, sym)
    hadamard!(K, kernelmatrix(κ₂, X, trans, uplo, sym))
end

# Returns the kernel matrix of X for a convex combination of two kernels
function kernelmatrix_sum{T<:FloatingPoint}(a₁::T, κ₁::StandardKernel{T}, a₂::T, 
                                            κ₂::StandardKernel{T}, X::Matrix{T},
                                            trans::Char = 'N', uplo::Char = 'U', 
                                            sym::Bool = true)
    K = kernelmatrix_scaled(a₁, κ₁, X, trans, uplo, sym)
    BLAS.axpy!(length(K), a₂, kernelmatrix(κ₂, X, trans, uplo, sym), 1, K, 1)
end

function kernelmatrix{T<:FloatingPoint}(ψ::ScaledKernel{T}, X::Matrix{T}, trans::Char = 'N',
                                        uplo::Char = 'U', sym::Bool = true)
    kernelmatrix_scaled(ψ.a, ψ.k, X, trans, uplo, sym)
end

function kernelmatrix{T<:FloatingPoint}(ψ::KernelProduct{T}, X::Matrix{T}, trans::Char = 'N',
                                        uplo::Char = 'U', sym::Bool = true)
    kernelmatrix_product(ψ.a, ψ.k1, ψ.k2, X, trans, uplo, sym)
end

function kernelmatrix{T<:FloatingPoint}(ψ::KernelSum{T}, X::Matrix{T}, trans::Char = 'N',
                                        uplo::Char = 'U', sym::Bool = true)
    kernelmatrix_sum(ψ.a1, ψ.k1, ψ.a2, ψ.k2, X, trans, uplo, sym)
end

function kernelmatrix{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T},
                                        trans::Char = 'N')
    idx = trans == 'N' ? 1 : 2
    n = size(X, idx)
    m = size(Y, idx)
    idx = trans == 'N' ? 2 : 1
    if size(X, idx) != size(Y, idx)
        throw(ArgumentError(
            "X and Y do not have the same number of " * trans == 'N' ? "rows." : "columns."
        ))
    end
    K = Array(T, n, m)
    if trans == 'N'
        for j = 1:m 
            for i = 1:n
                K[i,j] = kernel(κ, X[i,:], Y[j,:])
            end
        end
    else
        for j = 1:m 
            for i = 1:n
                K[i,j] = kernel(κ, X[:,i], Y[:,j])
            end
        end
    end
    K
end

function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::StandardKernel{T}, X::Matrix{T}, 
                                               Y::Matrix{T}, trans::Char = 'N')
    K = kernelmatrix(κ, X, Y, trans)
    a == one(T) ? K : BLAS.scal!(length(K), a, K, 1)
end

function kernelmatrix_product{T<:FloatingPoint}(a::T, κ₁::StandardKernel{T},
                                                κ₂::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T},
                                                trans::Char = 'N')
    K = kernelmatrix_scaled(a, κ₁, X, Y, trans)
    hadamard!(K, kernelmatrix(κ₂, X, Y, trans))
end

function kernelmatrix_sum{T<:FloatingPoint}(a₁::T, κ₁::StandardKernel{T}, a₂::T, 
                                            κ₂::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T},
                                            trans::Char = 'N')
    K = kernelmatrix_scaled(a₁, κ₁, X, Y, trans)
    BLAS.axpy!(length(K), a₂, kernelmatrix(κ₂, X, Y, trans), 1, K, 1)
end

function kernelmatrix{T<:FloatingPoint}(ψ::ScaledKernel{T}, X::Matrix{T}, Y::Matrix{T},
                                        trans::Char = 'N')
    kernelmatrix_scaled(ψ.a, ψ.k, X, Y, trans)
end
function kernelmatrix{T<:FloatingPoint}(ψ::KernelProduct{T}, X::Matrix{T}, Y::Matrix{T},
                                        trans::Char = 'N')
    kernelmatrix_product(ψ.a, ψ.k1, ψ.k2, X, Y, trans)
end
function kernelmatrix{T<:FloatingPoint}(ψ::KernelSum{T}, X::Matrix{T}, Y::Matrix{T},
                                        trans::Char = 'N')
    kernelmatrix_sum(ψ.a1, ψ.k1, ψ.a2, ψ.k2, X, Y, trans)
end


#==========================================================================
  Optimized kernel matrix functions for Euclidean distance and scalar
  product kernels
==========================================================================#

for (kernelobject, gramian) in ((:EuclideanDistanceKernel, :lagged_gramian_matrix),
                                (:ScalarProductKernel, :gramian_matrix))
    @eval begin

        # Kernelize a gramian matrix by transforming each element using the scalar kernel function
        function kernelize_gramian!{T<:FloatingPoint}(κ::$kernelobject{T}, G::Array{T})
            @inbounds for i = 1:length(G)
                G[i] = kernelize_scalar(κ, G[i])
            end
            G
        end

        # Kernelize a square gramian by only transforming the upper or lower triangle
        function kernelize_gramian!{T<:FloatingPoint}(κ::$kernelobject{T}, G::Array{T}, uplo::Char,
                                                      sym::Bool = true)
            n = size(G, 1)
            n == size(G, 2) || throw(ArgumentError("Gramian matrix must be square."))
            @inbounds for j = 1:n
                for i = uplo == 'U' ? (1:j) : (j:n)
                    G[i,j] = kernelize_scalar(κ, G[i,j])
                end 
            end
            sym ? (uplo == 'U' ? syml!(G) : symu!(G)) : G
        end

        # Returns kernel matrix of X using BLAS where possible
        function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T}, 
                                                trans::Char = 'N', uplo::Char = 'U', 
                                                sym::Bool = true)
            G = $gramian(X, trans, uplo, false)
            kernelize_gramian!(κ, G, uplo, sym)
        end

        # Returns scaled kernel matrix of X using BLAS where possible
        function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T},
                                                       trans::Char = 'N', uplo::Char = 'U',
                                                       sym::Bool = true)
            K = kernelmatrix(κ, X, trans, uplo, false)
            a == one(T) ? K : BLAS.scal!(length(K), a, K, 1)
            sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K
        end

        # Returns the kernel matrix of X for the product of two kernels using BLAS where possible
        function kernelmatrix_product{T<:FloatingPoint}(a::T, κ₁::$kernelobject{T}, 
                                                        κ₂::$kernelobject{T}, X::Matrix{T}, 
                                                        trans::Char = 'N', uplo::Char = 'U', 
                                                        sym::Bool = true)
            G = $gramian(X, trans, uplo, false)
            K = kernelize_gramian!(κ₁, copy(G), uplo, false)
            hadamard!(K, kernelize_gramian!(κ₂, G, uplo, false), uplo, false)
            if a != one(T)
                BLAS.scal!(length(K), a, K, 1)
            end
            sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K           
        end

        # Returns the kernel matrix of X for the sum of two kernels using BLAS where possible
        function kernelmatrix_sum{T<:FloatingPoint}(a₁::T, κ₁::$kernelobject{T}, a₂::T, 
                                                    κ₂::$kernelobject{T}, X::Matrix{T}, 
                                                    trans::Char = 'N', uplo::Char = 'U',
                                                    sym::Bool = true)
            G = $gramian(X, trans, uplo, false)
            K = kernelize_gramian!(κ₁, copy(G), uplo, false)
            n = length(K)
            if a₁ != one(T) 
                BLAS.scal!(n, a₁, K, 1)
            end
            BLAS.axpy!(n, a₂, kernelize_gramian!(κ₂, G, uplo, false), 1, K, 1)
            sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K 
        end

        # FIX BELOW
        function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T})
            G::Matrix{T} = $gramian(X, Y)
            K::Matrix{T} = kernelize_gramian!(κ, G)
        end

        function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T},
                                                       Y::Matrix{T})
            G::Matrix{T} = $gramian(X, Y)
            K::Matrix{T} = kernelize_gramian!(κ, G)
            if a != one(T) BLAS.scal!(length(K), a, K, 1) end
            K
        end

        function kernelmatrix_product{T<:FloatingPoint}(a::T, κ₁::$kernelobject{T}, 
                                                        κ₂::$kernelobject{T}, X::Matrix{T},
                                                        Y::Matrix{T})
            G::Matrix{T} = $gramian(X, Y)
            K::Matrix{T} = kernelize_gramian!(κ₁, copy(G))
            if a != one(T) BLAS.scal!(length(K), a, K, 1) end
            hadamard!(K, kernelize_gramian!(κ₂, G))
        end

        function kernelmatrix_sum{T<:FloatingPoint}(a₁::T, κ₁::$kernelobject{T}, a₂::T, 
                                                    κ₂::$kernelobject{T}, X::Matrix{T}, 
                                                    Y::Matrix{T})
            G::Matrix{T} = $gramian(X, Y)
            K::Matrix{T} = kernelize_gramian!(κ₁, copy(G))
            n = length(K)
            if a₁ != one(T) BLAS.scal!(n, a₁, K, 1) end
            BLAS.axpy!(n, a₂, kernelize_gramian!(κ₂, G), 1, K, 1)
        end
    end
end


#==========================================================================
  Optimized kernel matrix functions for Separable kernels
==========================================================================#

for kernelobject in (:MercerSigmoidKernel,)
    @eval begin

        function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T})
            K::Matrix{T} = BLAS.syrk('U', 'N', a, kernelize_array!(κ, copy(X)))
            syml!(K)
        end

        function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T})
            kernelmatrix_scaled(one(T), κ, X)
        end

        function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T},
                                                       Y::Matrix{T})
            K::Array{T} = BLAS.gemm('N', 'T', a, kernelize_array!(κ, copy(X)), 
                                                 kernelize_array!(κ, copy(Y)))
        end

        function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T})
            kernelmatrix_scaled(one(T), κ, X, Y)
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
    C::Matrix{T} = kernelmatrix(κ, X, X[sₓ,:])
    P::Matrix{T} = pinv_semiposdef!(C[sₓ,:])  # P'P = W = pinv(X[sₓ,sₓ]) 
    PCᵀ::Matrix{T} = BLAS.gemm('N', 'T', P, C)
    K = BLAS.syrk('U', 'T', one(T), PCᵀ)
    syml!(K)::Matrix{T}
end
