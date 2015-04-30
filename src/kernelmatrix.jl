#===================================================================================================
  Kernel Matrices
===================================================================================================#


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

# Returns the upper right corner of the gramian matrix of [Xᵀ Yᵀ]ᵀ or [X Y]
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

# Returns the kernel matrix of [Xᵀ Xᵀ]ᵀ or [X X]
#   If trans = 'N' then K = ϕ(X)ϕ(X)ᵀ (X is a design matrix)
#   If trans = 'T' then K = ϕ(X)ᵀϕ(X) (X is a transposed design matrix)
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

# Returns the upper right corner of the kernel matrix of [Xᵀ Yᵀ]ᵀ or [X Y]
#   If trans = 'N' then K = ϕ(X)ϕ(Y)ᵀ (X and Y are design matrices)
#   If trans = 'T' then K = ϕ(X)ᵀϕ(Y) (X and Y are transposed design matrices)
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
		G[i] = kernelize(κ, G[i])
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
		    G[i,j] = kernelize(κ, G[i,j])
		end 
	    end
	    sym ? (uplo == 'U' ? syml!(G) : symu!(G)) : G
	end

	# Returns kernel matrix of X using BLAS where possible
	function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T}, 
						trans::Char = 'N', uplo::Char = 'U', 
						sym::Bool = true)
	    G = $gramian(X, trans, uplo, false)
	    K = kernelize_gramian!(κ, G, uplo, sym)
	    sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K
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

	# Returns the kernel matrix of X and Y using BLAS where possible
	function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T},
						trans::Char = 'N')
	    G = $gramian(X, Y, trans)
	    kernelize_gramian!(κ, G)
	end

	# Returns the scaled kernel matrix of X and Y using BLAS where possible
	function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T},
						       Y::Matrix{T}, trans::Char = 'N')
	    G = $gramian(X, Y, trans)
	    K = kernelize_gramian!(κ, G)
	    a == one(T) ? K : BLAS.scal!(length(K), a, K, 1)
	end

	# Returns the kernel matrix of X and Y for the product of two kernels using BLAS 
	function kernelmatrix_product{T<:FloatingPoint}(a::T, κ₁::$kernelobject{T}, 
							κ₂::$kernelobject{T}, X::Matrix{T},
							Y::Matrix{T}, trans::Char = 'N')
	    G = $gramian(X, Y, trans)
	    K = kernelize_gramian!(κ₁, copy(G))
	    if a != one(T) 
		BLAS.scal!(length(K), a, K, 1) 
	    end
	    hadamard!(K, kernelize_gramian!(κ₂, G))
	end

	# Returns the kernel matrix of X and Y for the product of two kernels using BLAS 
	function kernelmatrix_sum{T<:FloatingPoint}(a₁::T, κ₁::$kernelobject{T}, a₂::T, 
						    κ₂::$kernelobject{T}, X::Matrix{T}, 
						    Y::Matrix{T}, trans::Char = 'N')
	    G = $gramian(X, Y, trans)
	    K = kernelize_gramian!(κ₁, copy(G))
	    n = length(K)
	    if a₁ != one(T) 
		BLAS.scal!(n, a₁, K, 1) 
	    end
	    BLAS.axpy!(n, a₂, kernelize_gramian!(κ₂, G), 1, K, 1)
	end

    end
end


#==========================================================================
	  Optimized kernel matrix functions for Separable kernels
==========================================================================#

for kernelobject in (:SeparableKernel,)
    @eval begin

	function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T},
						       trans::Char = 'N', uplo::Char = 'U',
						       sym::Bool = true)
	    K = BLAS.syrk(uplo, trans, a, kernelize_array!(κ, copy(X)))
	    sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K
	end

	function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T}, 
						trans::Char = 'N', uplo::Char = 'U',
						sym::Bool = true)
	    kernelmatrix_scaled(one(T), κ, X, )
	end

        function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T},
                                                       Y::Matrix{T}, trans::Char = 'N')
            K = BLAS.gemm(trans, trans == 'N' ? 'T' : 'N', a, kernelize_array!(κ, copy(X)), 
                                                              kernelize_array!(κ, copy(Y)))
        end

        function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T},
                                                trans::Char = 'N')
            kernelmatrix_scaled(one(T), κ, X, Y, trans)
        end

    end
end
