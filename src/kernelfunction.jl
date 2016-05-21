#===================================================================================================
  Generic kernel function
===================================================================================================#

call{T}(κ::Kernel{T}, x::T, y::T)                 = kernel(κ, x, y)
call{T}(κ::Kernel{T}, x::Vector{T}, y::Vector{T}) = kernel(κ, x, y)

call{T}(κ::Kernel{T}, X::Matrix{T})               = kernelmatrix(κ, X)
call{T}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}) = kernelmatrix(κ, X, Y)

kernel{T}(κ::PairwiseKernel{T}, x::T, y::T) = pairwise(κ, x, y)
kernel{T}(κ::PairwiseKernel{T}, x::AbstractVector{T}, y::AbstractVector{T}) = pairwise(κ, x, y)

kernel{T}(κ::KernelComposition{T}, x::T, y::T) = phi(κ.phi, pairwise(κ.k, x, y))
function kernel{T}(κ::KernelComposition{T}, x::AbstractVector{T}, y::AbstractVector{T})
    phi(κ.phi, pairwise(κ.k, x, y))
end


#==========================================================================
  Generic kernelmatrix Functions
==========================================================================#

function phi_matrix!{T}(ϕ::CompositionClass{T}, K::AbstractMatrix{T})
    @inbounds for i in eachindex(K)
        K[i] = phi(ϕ, K[i])
    end
    K
end

function phi_symmetricmatrix!{T}(ϕ::CompositionClass{T}, K::AbstractMatrix{T})
    if !((n = size(K,1)) == size(K,2))
        throw(DimensionMismatch("Kernel matrix must be square."))
    end
    @inbounds for j = 1:n, i = (1:j)
        K[i,j] = phi(ϕ, K[i,j])
    end
    LinAlg.copytri!(K, 'U')
end

function kernelmatrix{T}(κ::Kernel{T}, X::AbstractMatrix{T}, is_rowmajor::Bool = true)
    scheme = is_rowmajor ? Val{:row} : Val{:col}
    kernelmatrix!(scheme, init_pairwise(scheme, X), κ, X)
end

function kernelmatrix{T}(
        κ::Kernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T},
        is_rowmajor::Bool = true
    )
    scheme = is_rowmajor ? Val{:row} : Val{:col}
    kernelmatrix!(scheme, init_pairwise(scheme, X, Y), κ, X, Y)
end


#== Standard Kernel ==#

function kernelmatrix!{T}(v::DataType, K::Matrix{T}, κ::StandardKernel{T}, X::AbstractMatrix{T})
    pairwise!(v, K, κ, X)
end

function kernelmatrix!{T}(
        v::DataType,
        K::Matrix{T},
        κ::StandardKernel{T},
        X::AbstractMatrix{T}
    )
    pairwise!(v, K, κ, X)
end


#== Composition Kernel ==#

function kernelmatrix!{T}(v::DataType, K::Matrix{T}, κ::KernelComposition{T}, X::AbstractMatrix{T})
    pairwise!(v, K, κ.kappa, X)
    phi_symmetricmatrix!(κ.phi, K)
end

function kernelmatrix!{T}(
        v::DataType,
        K::Matrix{T},
        κ::KernelComposition{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwise!(v, K, κ.kappa, X, Y)
    phi_matrix!(κ.phi, K)
end


#== Kernel Operation ==#

for (kernel_object, scalar_op, identity, scalar) in (
        (:KernelProduct, :*, :1, :a),
        (:KernelSum,     :+, :0, :c)
    )
    @eval begin
        function kernelmatrix!{T}(
                v::Union{Type{Val{:row}},Type{Val{:col}}},
                K::Matrix{T},
                κ::$kernel_object{T},
                X::AbstractMatrix{T}
            )
            kernelmatrix!(v, K, κ.kappa1, X)
            broadcast!($scalar_op, K, kernelmatrix(v, similar(K), κ.kappa2, X))
            κ.$scalar == $identity ? K : broadcast!($scalar_op, K, κ.$scalar)
        end

        function kernelmatrix!{T}(
                v::Union{Type{Val{:row}},Type{Val{:col}}},
                K::Matrix{T},
                κ::$kernel_object{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T}
            )
            kernelmatrix!(v, K, κ.kappa1, X, Y)
            broadcast!($scalar_op, K, kernelmatrix(v, similar(K), κ.kappa2, X, Y))
            κ.$scalar == $identity ? K : broadcast!($scalar_op, K, κ.$scalar)
        end
    end
end




#==========================================================================
  Kernel Matrix Transformation
==========================================================================

# Centralize a kernel matrix K
function centerkernelmatrix!{T<:AbstractFloat}(K::Matrix{T})
    (n = size(K, 1)) == size(K, 2) || throw(DimensionMismatch("Kernel matrix must be square"))
    μ_row = zeros(T,n)
    μ = zero(T)
    @inbounds for j = 1:n
        @simd for i = 1:n
            μ_row[j] += K[i,j]
        end
        μ += μ_row[j]
        μ_row[j] /= n
    end
    μ /= n^2
    @inbounds for j = 1:n
        @simd for i = 1:n
            K[i,j] += μ - μ_row[i] - μ_row[j]
        end
    end
    K
end
centerkernelmatrix{T<:AbstractFloat}(K::Matrix{T}) = centerkernelmatrix!(copy(K))

==========================================================================
  Kernel Operation Matrix Functions
==========================================================================


for (kernel_object, matrix_op, array_op, identity, scalar) in (
        (:KernelProduct, :matrix_prod!, :scale!,     :1, :a),
        (:KernelSum,     :matrix_sum!,  :translate!, :0, :c)
    )
    @eval begin

        function kernelmatrix!{T<:AbstractFloat}(K::Matrix{T}, κ::$kernel_object{T}, X::Matrix{T}, 
                                                 is_trans::Bool, store_upper::Bool, 
                                                 symmetrize::Bool)
            nk = length(κ.k)
            kernelmatrix!(K, κ.k[1], X, is_trans, store_upper, false)
            if nk > 1
                K_factor = similar(K)
                for i = 2:nk
                    ($matrix_op)(K, kernelmatrix!(K_factor, κ.k[i], X, is_trans, store_upper, false), store_upper, false)
                end
            end
            K = symmetrize ? (store_upper ? syml!(K) : symu!(K)) : K
            κ.$scalar == $identity ? K : ($array_op)(κ.$scalar, K)
        end

        function kernelmatrix!{T<:AbstractFloat}(K::Matrix{T}, κ::$kernel_object{T}, X::Matrix{T},
                                                 Y::Matrix{T}, is_trans::Bool)
            nk = length(κ.k)
            kernelmatrix!(K, κ.k[1], X, Y, is_trans)
            if nk > 1
                K_factor = similar(K)
                for i = 2:nk
                    ($matrix_op)(K, kernelmatrix!(K_factor, κ.k[i], X, Y, is_trans))
                end
            end
            κ.$scalar == $identity ? K : ($array_op)(κ.$scalar, K)
        end

    end
end

=#
