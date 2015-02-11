# syml!: Symmetrize the lower half of matrix S using the upper half of S
function syml!(S::Matrix)
    p = size(S, 1)
    p == size(S, 2) || error("S ∈ ℝ$p×$(size(S, 2)) should be square")
    if p > 1 
        for j = 1:p-1 
            for i = (j + 1):p 
                S[i, j] = S[j, i]
            end
        end
    end
    return S
end
syml(S::Matrix) = syml!(copy(S))


# kernelmatrix:
#   Returns the kernel (Gramian) matrix K of X for mapping ϕ
function kernel_matrix{T<:FloatingPoint}(X::Matrix{T}, κ::MercerKernel = LinearKernel(); symmetrize::Bool = true)
    k = kernel(κ)
    n = size(X, 1)
    K = Array(T, n, n) # Kᵩ = XᵩXᵗᵩ
    for i = 1:n 
        for j = i:n
            K[i,j] = k(X[i,:], X[j,:])
        end 
    end
    return symmetrize ? syml!(K) : K
end


# kernelmatrix:
#   Returns the upper right corner kernel (Gramian) matrix K of [Xᵗ,Zᵗ]ᵗ
function kernelmatrix{T<:FloatingPoint}(X::Matrix{T}, Z::Matrix{T}, κ::MercerKernel = LinearKernel())
    k = kernel(κ)
    n = size(X, 1)
    m = size(Z, 1)
    size(X, 2) == size(Z, 2) || error("X ∈ ℝn×p and Z should be ∈ ℝm×p, but X ∈ ℝn×$(size(X, 2)) and Z ∈ ℝm×$(size(Z, 2)).")
    K = Array(T, n, m) # K = XᵩZᵗᵩ
    for j = 1:m 
        for i = 1:n
            K[i,j] = k(X[i,:], Z[j,:])
        end
    end
    return K
end

#===================================================================================================
  Kernel Matrix Functions
===================================================================================================#

# center_kernelmatrix!: Centralize a kernel matrix K
#	K := K - 1ₙ*K/n - K*1ₙ/n + 1ₙ*K*1ₙ/n^2
#	 • K is an n×n kernel matrix
#	 • 1ₙ is an n×n matrix of ones
function center_kernelmatrix!{T<:FloatingPoint}(K::Matrix{T})
	n = size(K,1)
	n == size(K,2) || error("Kernel matrix must be square")
	κ = sum(K,1)
	μₖ = sum(κ)/(convert(T,n)^2)
	BLAS.scal!(n,one(T)/convert(T,n),κ,1)
	return MATRIX.el_add!(MATRIX.col_add!(MATRIX.row_add!(K,-κ),-κ),μₖ)
end
center_kernelmatrix{T<:FloatingPoint}(K::Matrix{T}) = center_kernelmatrix!(copy(K))


