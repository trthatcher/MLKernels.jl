#===================================================================================================
  Kernel Matrices
===================================================================================================#

#==========================================================================
  Kernel Matrix Transformation
==========================================================================#

# Centralize a kernel matrix K
function center_kernelmatrix!{T<:FloatingPoint}(K::Matrix{T})
    (n = size(K, 1)) == size(K, 2) || error("Kernel matrix must be square")
    row_mean = sum(K, 1)
    element_mean = sum(row_mean) / (convert(T, n)^2)
    BLAS.scal!(n, one(T)/convert(T,n), row_mean, 1)
    ((K .- row_mean) .- row_mean') .+ element_mean
end
center_kernelmatrix{T<:FloatingPoint}(K::Matrix{T}) = center_kernelmatrix!(copy(K))


#==========================================================================
  Generic Kernel Matrix Functions
==========================================================================#

function kernelmatrix{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
    is_trans = trans == 'T'
    kernelmatrix!(init_gramian(X, is_trans), κ, X, is_trans, uplo == 'U', sym)
end

function kernelmatrix{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    is_trans = trans == 'T'
    kernelmatrix!(init_gramian(X, Y, is_trans), κ, X, Y, is_trans)
end


#==========================================================================
  Generic Simple Kernel Matrix Functions
==========================================================================#

# Returns the kernel matrix of [Xᵀ Xᵀ]ᵀ or [X X]
#   If trans == 'N' -> gramian(X) = XXᵀ (X is a design matrix IE rows are coordinates)
#            == 'T' -> gramian(X) = XᵀX (X is a transposed design matrix IE columns are coordinates)
function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::StandardKernel{T}, X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    n = size(X, is_trans ? 2 : 1)
    if size(K) != (n, n)
        throw(ArgumentError(string("X is ", description_matrix_size(X), ", but K is ", description_matrix_size(K), "; K must be $n×$n.")))
    end
    @transpose_access is_trans (X,) @inbounds for j = 1:n
        for i = is_upper ? (1:j) : (j:n)
            K[i,j] = kernel(κ, X[i,:], X[j,:])
        end 
    end
    sym ? (is_upper ? syml!(K) : symu!(K)) : K
end


# Returns the upper right corner of the kernel matrix of [Xᵀ Yᵀ]ᵀ or [X Y]
#   If trans == 'N' -> gramian(X) = XXᵀ (X is a design matrix IE rows are coordinates)
#            == 'T' -> gramian(X) = XᵀX (X is a transposed design matrix IE columns are coordinates)
function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
    n = size(X, is_trans ? 2 : 1)
    m = size(Y, is_trans ? 2 : 1)
    if (d = size(X, is_trans ? 1 : 2)) != size(Y, is_trans ? 1 : 2)
        throw(DimensionMismatch("X and Y do not have the same number of " * (is_trans ? "rows." : "columns.")))
    end
    if size(K) != (n, m)
        desc = is_trans ? "X is $d×$n and Y is $d×$m" : "X is $n×$d and Y is $m×$d"
        throw(ArgumentError(string(desc, ", but K is ", description_matrix_size(K), "; K must be $n×$m.")))
    end
    @transpose_access is_trans (X,Y) @inbounds for j = 1:m 
        for i = 1:n
            K[i,j] = kernel(κ, X[i,:], Y[j,:])
        end
    end
    K
end


#==========================================================================
  Generic Composite Kernel Matrix Functions
==========================================================================#

for (kernel_object, matrix_op, array_op, identity) in ((:KernelProduct, :matrix_prod!, :scale!, :1), (:KernelSum, :matrix_sum!, :translate!, :0))
    @eval begin

        function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::$kernel_object{T}, X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
            c = length(κ.k)
            kernelmatrix!(K, κ.k[1], X, is_trans, is_upper, false)
            if c > 1
                K_factor = safe_similar(K)
                for i = 2:c
                    ($matrix_op)(K, kernelmatrix!(K_factor, κ.k[i], X, is_trans, is_upper, false), is_upper, false)
                end
            end
            K = sym ? (is_upper ? syml!(K) : symu!(K)) : K
            κ.a == $identity ? K : ($array_op)(κ.a, K)
        end

        function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::$kernel_object{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
            c = length(κ.k)
            kernelmatrix!(K, κ.k[1], X, Y, is_trans)
            if c > 1
                K_factor = safe_similar(K)
                for i = 2:c
                    ($matrix_op)(K, kernelmatrix!(K_factor, κ.k[i], X, Y, is_trans))
                end
            end
            κ.a == $identity ? K : ($array_op)(κ.a, K)
        end

    end
end


#==========================================================================
  Optimized kernel matrix functions for Euclidean distance and scalar
  product kernels
==========================================================================#

# Kernelize a gramian matrix by transforming each element using the scalar kernel function
function kappa_gramian!{T<:FloatingPoint}(κ::Union(SquaredDistanceKernel{T}, ScalarProductKernel{T}), G::Array{T})
    @inbounds for i = 1:length(G)
        G[i] = kappa(κ, G[i])
    end
    G
end

# Kernelize a square gramian by only transforming the upper or lower triangle
function kappa_gramian!{T<:FloatingPoint}(κ::Union(SquaredDistanceKernel{T}, ScalarProductKernel{T}), G::Array{T}, is_upper::Bool, sym::Bool = true)
    (n = size(G, 1)) == size(G, 2) || throw(DimensionMismatch("Gramian matrix must be square."))
    @inbounds for j = 1:n
        for i = is_upper ? (1:j) : (j:n)
            G[i,j] = kappa(κ, G[i,j])
        end 
    end
    sym ? (is_upper ? syml!(G) : symu!(G)) : G
end

for (kernelobject, gramian) in ((:SquaredDistanceKernel, :sqdistmatrix!), 
                                (:ScalarProductKernel, :scprodmatrix!))
    @eval begin
    function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::$kernelobject{T}, X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
        $gramian(K, X, is_trans, is_upper, false)
        kappa_gramian!(κ, K, is_upper, sym)
    end

    function kernelmatrix!{T<:FloatingPoint,U<:$kernelobject}(K::Matrix{T}, κ::ARD{T,U}, X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
        $gramian(K, X, κ.w, is_trans, is_upper, false)
        kappa_gramian!(κ.k, K, is_upper, sym)
    end

    function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
        $gramian(K, X, Y, is_trans)
        kappa_gramian!(κ, K)
    end

    function kernelmatrix!{T<:FloatingPoint,U<:$kernelobject}(K::Matrix{T}, κ::ARD{T,U}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
        $gramian(K, X, Y, κ.w, is_trans)
        kappa_gramian!(κ.k, K)
    end

    end
end
