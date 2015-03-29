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
        for j = 1:(p - 1) 
            for i = (j + 1):p 
                S[i, j] = S[j, i]
            end
        end
    end
    return S
end
syml(S::Matrix) = syml!(copy(S))

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

# Overwrite A with the hadamard product of A and B. Returns A
function hadamard!{T<:FloatingPoint}(A::Matrix{T}, B::Matrix{T})
    size(A) == size(B) || error("Dimensions do not conform.")
    @inbounds for i = 1:length(A)
        A[i] *= B[i]
    end
    A
end


#==========================================================================
  Auxiliary Functions
==========================================================================#

# Calculate the gramian G = XXᵀ of matrix X
function gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, sym::Bool = true)
    G::Array{T} = BLAS.syrk('U', 'N', one(T), X)
    sym ? syml!(G) : G
end

# Calculate the upper right corner G = XYᵀ of the gramian of matrix [Xᵀ Yᵀ]ᵀ
function gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T})
    G::Array{T} = BLAS.gemm('N', 'T', one(T), X, Y)
end

# Calculates G such that Gij is the dot product of the difference of row i and j of matrix X
function lagged_gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, sym::Bool = true)
    n = size(X, 1)
    G::Array{T} = gramian_matrix(X, false)
    xᵀx = copy(vec(diag(G)))
    @inbounds for j = 1:n
        for i = 1:j
            G[i,j] = xᵀx[i] - convert(T, 2) * G[i,j] + xᵀx[j]
        end
    end
    sym ? syml!(G) : G
end

# Calculates the upper right corner G of the lagged gramian matrix of matrix [Xᵀ Yᵀ]ᵀ
function lagged_gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T})
    n = size(X, 1)
    m = size(Y, 1)
    xᵀx::Vector{T} = dot_rows(X)
    yᵀy::Vector{T} = dot_rows(Y)
    G::Array{T} = gramian_matrix(X, Y)
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

function init_approx{T<:FloatingPoint}(X::Matrix{T}, Sample::Array{Int}, kernel::Kernel = LinearKernel())
    k = kernelfunction(kernel)
    c = length(Sample)
    n = size(X, 1)
    Cᵀ = Array(T, n, c)
    for i = 1:n
        for j = 1:c
            Cᵀ[i,j] = k(X[i,:], X[Sample[j],:])
        end
    end
    W = pinv(Cᵀ[Sample,:])
    return Cᵀ * W * Cᵀ'
end

