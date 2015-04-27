#===================================================================================================
  Kernel Matrix Approximation
===================================================================================================#

#===================================================================================================
  Nystrom Method
===================================================================================================#

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
