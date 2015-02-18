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


#===================================================================================================
  Kernel Matrix Functions
===================================================================================================#

# kernelmatrix:
#   Returns the kernel (Gramian) matrix K of X for mapping ϕ
function gramian{T<:FloatingPoint}(X::Matrix{T}, kernel::MercerKernel = LinearKernel(), symmetrize::Bool = true)
    k = kernelfunction(kernel)
    n = size(X, 1)
    K = Array(T, n, n)
    for i = 1:n 
        for j = i:n
            K[i,j] = k(X[i,:], X[j,:])
        end 
    end
    return symmetrize ? syml!(K) : K
end


# kernelmatrix:
#   Returns the upper right corner kernel (Gramian) matrix K of [Xᵗ,Zᵗ]ᵗ
function gramian{T<:FloatingPoint}(X::Matrix{T}, Z::Matrix{T}, kernel::MercerKernel = LinearKernel())
    k = kernelfunction(kernel)
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


# center_kernelmatrix!: Centralize a kernel matrix K
#	K := K - 1ₙ*K/n - K*1ₙ/n + 1ₙ*K*1ₙ/n^2
function center_gramian!{T<:FloatingPoint}(K::Matrix{T})
	n = size(K, 1)
	n == size(K, 2) || error("Kernel matrix must be square")
	row_mean = sum(K, 1)
	element_mean = sum(κ)/(convert(T,n)^2)
	BLAS.scal!(n,one(T)/convert(T,n), κ, 1)
	return ((K .- row_mean) .- row_mean') .+ element_mean
end
center_gramian{T<:FloatingPoint}(K::Matrix{T}) = center_gramian!(copy(K))

#===================================================================================================
  Kernel Matrix Functions
===================================================================================================#

function init_approx{T<:FloatingPoint}(X::Matrix{T}, Sample::Array{Int}, kernel::MercerKernel = LinearKernel())
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

