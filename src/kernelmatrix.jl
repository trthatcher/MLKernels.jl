#===================================================================================================
  Kernel Matrices
===================================================================================================#

macro check_KXX_dims(is_trans, K, X, n)
    quote
        $n = size($X, is_trans ? 2 : 1)
        if size($K) != ($n, $n)
            desc = string(" but K is ", size($K,1), "×", size($K,2), "; K must be ", $n, "×", $n, ".")
            if $is_trans
                throw(ArgumentError(string("X is ", size($X,1), "×", $n, desc)))
            else
                throw(ArgumentError(string("X is ", $n, "×", size($X,1), desc)))
            end
        end
    end
end

macro check_KXY_dims(is_trans, K, X, n, Y, m, d)
    quote
        $n, $m = $is_trans ? (size($X, 2), size($Y,2)) : (size($X, 1), size($Y,1))    
        if ($d = size($X, is_trans ? 1 : 2)) != size($Y, $is_trans ? 1 : 2)
            throw(ArgumentError("X and Y do not have the same number of " * (is_trans ? "rows." : "columns.")))
        end
        if size($K) != ($n,$m)
            desc = string(" but K is ", size($K,1), "×", size($K,2), "; K must be ", $n, "×", $m, ".")
            if $is_trans
                throw(ArgumentError(string("X is ", $d, "×", $n, " and Y is ", $d, "×", $m, desc)))
            else
                throw(ArgumentError(string("X is ", $n, "×", $d, " and Y is ", $m, "×", $d, desc)))
            end
        end
    end
end

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
  Generic Simple Kernel Matrix Functions
==========================================================================#

# Returns the kernel matrix of [Xᵀ Xᵀ]ᵀ or [X X]
#   If trans == 'N' -> gramian(X) = XXᵀ (X is a design matrix IE rows are coordinates)
#            == 'T' -> gramian(X) = XᵀX (X is a transposed design matrix IE columns are coordinates)
function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::SimpleKernel{T}, X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    @check_KXX_dims(is_trans, K, X, n)
    @transpose_access is_trans (X,) @inbounds for j = 1:n
        for i = is_upper ? (1:j) : (j:n)
            K[i,j] = kernel(κ, X[i,:], X[j,:])
        end 
    end
    sym ? (is_upper ? syml!(K) : symu!(K)) : K
end

function kernelmatrix{T<:FloatingPoint}(κ::SimpleKernel{T}, X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
    is_trans = trans == 'T'
    n = size(X, is_trans ? 2 : 1)
    kernelmatrix!(Array(T, n, n), κ, X, is_trans, uplo == 'U', true)
end

# Returns the upper right corner of the kernel matrix of [Xᵀ Yᵀ]ᵀ or [X Y]
#   If trans == 'N' -> gramian(X) = XXᵀ (X is a design matrix IE rows are coordinates)
#            == 'T' -> gramian(X) = XᵀX (X is a transposed design matrix IE columns are coordinates)
function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::SimpleKernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
    @check_kernelmatrix_dims(is_trans, K, X, n, Y, m, d)
    @transpose_access is_trans (X,Y) @inbounds for j = 1:m 
        for i = 1:n
            K[i,j] = kernel(κ, X[i,:], Y[j,:])
        end
    end
    K
end

function kernelmatrix{T<:FloatingPoint}(κ::SimpleKernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    is_trans = trans == 'T'  # True if columns are observations
    n, m = is_trans ? (size(X, 2), size(Y,2)) : (size(X, 1), size(Y,1))
    kernelmatrix!(Array(T, n, m), κ, X, Y, is_trans)
end


#==========================================================================
  Generic Kernel Matrix Product Functions
==========================================================================#

function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::KernelProduct{T}, X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    @check_KXX_dims(is_trans, K, X, n)
    c = length(κ.k)
    kernelmatrix!(K, κ.k[1], X, is_trans, is_upper, false)
    if c > 1
        K_factor = Array(T, n, n)
        for i = 2:c
            hadamard!(K, kernelmatrix!(K_factor, κ.k[i], X, is_trans, is_upper, false))
        end
    end
    κ.a == 1 ? K : scale!(a, K)
    sym ? (is_upper ? syml!(K) : symu!(K)) : K
end

function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::KernelProduct{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
    @check_kernelmatrix_dims(is_trans, K, X, n, Y, m, d)
    c = length(κ.k)
    kernelmatrix!(K, κ.k[1], X, Y, is_trans)
    if c > 1
        K_factor = Array(T, n, m)
        for i = 2:c
            hadamard!(K, kernelmatrix!(K_factor, κ.k[i], X, Y, is_trans))
        end
    end
    κ.a == 1 ? K : scale!(a, K)
end

#==========================================================================
  Generic Kernel Matrix Sum Functions
==========================================================================#

function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::KernelSum{T}, X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = true, sym::Bool = true)
    @check_KXX_dims(is_trans, K, X, n)
    c = length(κ.k)
    kernelmatrix!(K, κ.k[1], X, is_trans, is_upper, false)
    if c > 1
        K_factor = Array(T, n, n)
        for i = 2:c
            axpy!(one(T), kernelmatrix!(K_factor, κ.k[i], X, is_trans, is_upper, false), K)
        end
    end
    sym ? (is_upper ? syml!(K) : symu!(K)) : K
end

function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::KernelSum{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool = false)
    @check_kernelmatrix_dims(is_trans, K, X, n, Y, m, d)
    c = length(κ.k)
    kernelmatrix!(K, κ.k[1], X, Y, is_trans)
    if c > 1
        K_factor = Array(T, n, m)
        for i = 2:c
            axpy!(one(T), kernelmatrix!(K_factor, κ.k[i], X, Y, is_trans), K)
        end
    end
    K
end


#==========================================================================
  Optimized kernel matrix functions for Euclidean distance and scalar
  product kernels
==========================================================================#

for (kernelobject, gramian) in ((:SquaredDistanceKernel, :sqdistmatrix), 
                                (:ScalarProductKernel, :scprodmatrix))
    @eval begin

    # Kernelize a gramian matrix by transforming each element using the scalar kernel function
    function kappa_gramian!{T<:FloatingPoint}(κ::$kernelobject{T}, G::Array{T})
        @inbounds for i = 1:length(G)
            G[i] = kappa(κ, G[i])
        end
        G
    end

    # Kernelize a square gramian by only transforming the upper or lower triangle
    function kappa_gramian!{T<:FloatingPoint}(κ::$kernelobject{T}, G::Array{T}, is_upper::Bool = true, sym::Bool = true)
        (n = size(G, 1)) == size(G, 2) || throw(ArgumentError("Gramian matrix must be square."))
        @inbounds for j = 1:n
            for i = uplo == 'U' ? (1:j) : (j:n)
                G[i,j] = kappa(κ, G[i,j])
            end 
        end
        sym ? (uplo == 'U' ? syml!(G) : symu!(G)) : G
    end

    function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::$kernelobject{T}, X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
        fill!(K, zero(T))
        G = $gramian(X, trans, uplo, false)
        K = kappa_gramian!(κ, G, uplo, sym)
        sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K
    end

    function kernelmatrix!{T<:FloatingPoint,U<:$kernelobject}(K::Matrix{T}, κ::ARD{T,U}, X::Matrix{T}, is_trans::Bool = false, is_upper::Bool = false, sym::Bool = true)
        fill!(K, zero(T))
        G = $gramian(X, trans, uplo, false)
        K = kappa_gramian!(κ, G, uplo, sym)
        sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K
    end
    



    # Returns kernel matrix of X using BLAS where possible

    # Returns scaled kernel matrix of X using BLAS where possible
    function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
        K = kernelmatrix(κ, X, trans, uplo, false)
        a == one(T) ? K : BLAS.scal!(length(K), a, K, 1)
        sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K
    end

    # Returns the kernel matrix of X for the product of two kernels using BLAS where possible
    function kernelmatrix_product{T<:FloatingPoint}(a::T, κ₁::$kernelobject{T}, κ₂::$kernelobject{T}, X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
        G = $gramian(X, trans, uplo, false)
        K = kappa_gramian!(κ₁, copy(G), uplo, false)
        hadamard!(K, kappa_gramian!(κ₂, G, uplo, false), uplo, false)
        if a != one(T)
            BLAS.scal!(length(K), a, K, 1)
        end
        sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K           
    end

    # Returns the kernel matrix of X for the sum of two kernels using BLAS where possible
    function kernelmatrix_sum{T<:FloatingPoint}(a₁::T, κ₁::$kernelobject{T}, a₂::T, κ₂::$kernelobject{T}, X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
        G = $gramian(X, trans, uplo, false)
        K = kappa_gramian!(κ₁, copy(G), uplo, false)
        n = length(K)
        if a₁ != one(T) 
            BLAS.scal!(n, a₁, K, 1)
        end
        BLAS.axpy!(n, a₂, kappa_gramian!(κ₂, G, uplo, false), 1, K, 1)
        sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K 
    end

    # Returns the kernel matrix of X and Y using BLAS where possible
    function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
        G = $gramian(X, Y, trans)
        kappa_gramian!(κ, G)
    end

    # Returns the scaled kernel matrix of X and Y using BLAS where possible
    function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
        G = $gramian(X, Y, trans)
        K = kappa_gramian!(κ, G)
        a == one(T) ? K : BLAS.scal!(length(K), a, K, 1)
    end

    # Returns the kernel matrix of X and Y for the product of two kernels using BLAS 
    function kernelmatrix_product{T<:FloatingPoint}(a::T, κ₁::$kernelobject{T}, κ₂::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
        G = $gramian(X, Y, trans)
        K = kappa_gramian!(κ₁, copy(G))
        if a != one(T) 
            BLAS.scal!(length(K), a, K, 1) 
        end
        hadamard!(K, kappa_gramian!(κ₂, G))
    end

    # Returns the kernel matrix of X and Y for the product of two kernels using BLAS 
    function kernelmatrix_sum{T<:FloatingPoint}(a₁::T, κ₁::$kernelobject{T}, a₂::T, κ₂::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
        G = $gramian(X, Y, trans)
        K = kappa_gramian!(κ₁, copy(G))
        n = length(K)
        if a₁ != one(T) 
            BLAS.scal!(n, a₁, K, 1) 
        end
        BLAS.axpy!(n, a₂, kappa_gramian!(κ₂, G), 1, K, 1)
    end

    end
end


#==========================================================================
      Optimized kernel matrix functions for Separable kernels
==========================================================================#

for kernelobject in (:SeparableKernel,)
    @eval begin

    function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
        K = BLAS.syrk(uplo, trans, a, kappa_array!(κ, copy(X)))
        sym ? (uplo == 'U' ? syml!(K) : symu!(K)) : K
    end

    function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true)
        kernelmatrix_scaled(one(T), κ, X, )
    end

    function kernelmatrix_scaled{T<:FloatingPoint}(a::T, κ::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
        BLAS.gemm(trans, trans == 'N' ? 'T' : 'N', a, kappa_array!(κ, copy(X)), kappa_array!(κ, copy(Y)))
    end

    function kernelmatrix{T<:FloatingPoint}(κ::$kernelobject{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
        kernelmatrix_scaled(one(T), κ, X, Y, trans)
    end

    end
end
