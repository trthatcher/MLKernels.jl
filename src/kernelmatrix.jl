#===================================================================================================
  Auxiliary Functions
===================================================================================================#

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

function gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, sym::Bool = true)
    G::Array{T} = BLAS.syrk('U', 'N', one(T), X)
    sym ? syml!(G) : G
end

function gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T})
    G::Array{T} = BLAS.gemm('N', 'T', one(T), X, Y)
end

function lagged_gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T})
    n = size(X, 1)
    m = size(Y, 1)
    xᵗx::Vector{T} = vec(sum(X .* X, 1))
    yᵗy::Vector{T} = vec(sum(Y .* Y, 1))
    G::Array{T} = gramian_matrix(X, Y)
    @inbounds for j = 1:m
        for i = 1:n
            G[i,j] = xᵗx[i] - convert(T, -2) * G[i,j] + yᵗy[j]
        end
    end
    G
end

function lagged_gramian_matrix{T<:FloatingPoint}(X::Matrix{T}, sym::Bool = true)
    n = size(X, 1)
    G::Array{T} = gramian_matrix(X, false)
    xᵗx = copy(vec(diag(G)))
    @inbounds for j = 1:n
        for i = 1:n
            G[i,j] = xᵗx[i] - convert(T, -2) * G[i,j] + xᵗx[j]
        end
    end
    sym ? syml!(G) : G
end

for kernel in (:EuclideanDistanceKernel, :ScalarProductKernel)
    @eval begin
        function kernelize_gramian!{T<:FloatingPoint}(G::Array{T}, κ::$kernel{T})
            @inbounds for i = 1:length(G)
                G[i] = kernelize_scalar(κ, G[i])
            end
            G
        end

        function kernelize_gramian_scaled!{T<:FloatingPoint}(G::Array{T}, a::T, κ::$kernel{T})
            @inbounds for i = 1:length(G)
                G[i] = a * kernelize_scalar(κ, G[i])
            end
            G
        end

        function kernelize_gramian_product!{T<:FloatingPoint}(G::Array{T}, a::T, κ₁::$Kernel{T},
                                                              κ₂::$Kernel{T})
            @inbounds for i = 1:length(G)
                G[i] = a*kernelize_scalar(κ₁, G[i])*kernelize_scalar(κ₂, G[i])
            end
            G
        end

        function kernelize_gramian_sum!{T<:FloatingPoint}(G::Array{T}, a₁::T, κ₁::$Kernel{T}, a₂::T,
                                                          κ₂::$Kernel{T})
            @inbounds for i = 1:length(G)
                G[i] = a₁*kernelize_scalar(κ₁, G[i]) + a₂*kernelize_scalar(κ₂, G[i])
            end
            G
        end
    end
end




#===================================================================================================
  Kernel Matrix Functions
===================================================================================================#


function kernel_matrix{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T})
    n = size(X, 1)
    K = Array(T, n, n)
    @inbounds for i = 1:n 
        for j = i:n
            K[i,j] = kernel_function(κ, vec(X[i,:]), vec(X[j,:]))
        end 
    end
    syml!(K)
end







for (kernel, gramian) in ((:EuclideanDistanceKernel, :gramian_matrix),
                          (:ScalarProductKernel, :lagged_gramian_matrix))
    @eval begin
        function kernel_matrix{T<:FloatingPoint}(κ::$kernel{T}, X::Matrix{T})
            G::Matrix{T} = $gramian(X)
            K::Matrix{T} = kernelize_gramian!(G, κ)
        end

        function kernel_matrix_scaled{T<:FloatingPoint}(a::T, κ::$Kernel{T}, X::Array{T})
            G::Matrix{T} = $gramian(X)
            K::Matrix{T} = kernelize_gramian_sum!(G, a, κ)
        end

        function kernel_matrix_product{T<:FloatingPoint}(a::T, κ₁::$Kernel{T}, κ₂::$Kernel{T},
                                                         X::Array{T})
            G::Matrix{T} = $gramian(X)
            K::Matrix{T} = kernelize_gramian_sum!(G, a, κ₁, κ₂)
        end

        function kernel_matrix_sum{T<:FloatingPoint}(a₁::T, κ₁::$Kernel{T}, a₂::T, κ₂::$Kernel{T},
                                                     X::Array{T})
            G::Matrix{T} = $gramian(X)
            K::Matrix{T} = kernelize_gramian_sum!(G, a₁, κ₁, a₂, κ₂)
        end

    end
end




#function apply_function{T<:FloatingPoint}(X::Matrix{T}


# Returns the kernel (Gramian) matrix K of data matrix X for mapping ϕ
#=
function kernel_matrix{T<:FloatingPoint}(X::Matrix{T}, κ::Kernel = LinearKernel(), sym::Bool = true)
    k = kernel_function(κ)
    n = size(X, 1)
    K = Array(T, n, n)
    for i = 1:n 
        for j = i:n
            K[i,j] = k(X[i,:], X[j,:])  # @inbounds?
        end 
    end
    sym ? syml!(K) : K
end
=#

# Returns the upper right corner kernel (Gramian) matrix K of data matrix [Xᵗ,Zᵗ]ᵗ
#=
function kernel_matrix{T<:FloatingPoint}(X::Matrix{T}, Z::Matrix{T}, κ::Kernel = LinearKernel())
    k = kernel_function(κ)
    n = size(X, 1)
    m = size(Z, 1)
    size(X, 2) == size(Z, 2) || error("X ∈ ℝn×p and Z should be ∈ ℝm×p, but X ∈ " * (
                                      "ℝn×$(size(X, 2)) and Z ∈ ℝm×$(size(Z, 2))."))
    K = Array(T, n, m)
    for j = 1:m 
        for i = 1:n
            K[i,j] = k(X[i,:], Z[j,:])  # @inbounds?
        end
    end
    return K
end
=#

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


#===================================================================================================
  Kernel Matrix Functions
===================================================================================================#

function init_approx{T<:FloatingPoint}(X::Matrix{T}, Sample::Array{Int}, kernel::Kernel = LinearKernel())
    k = kernelfunction(kernel)
    c = length(Sample)
    n = size(X, 1)
    Cᵗ = Array(T, n, c)
    for i = 1:n
        for j = 1:c
            Cᵗ[i,j] = k(X[i,:], X[Sample[j],:])
        end
    end
    W = pinv(Cᵗ[Sample,:])
    return Cᵗ * W * Cᵗ'
end

