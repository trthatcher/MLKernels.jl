#===================================================================================================
  Generic kernel function
===================================================================================================#

call{T}(κ::Kernel{T}, x::T, y::T) = kernel(κ, x, y)
call{T}(κ::Kernel{T}, x::AbstractArray{T}, y::AbstractArray{T}) = kernel(κ, x, y)

kernel{T}(κ::StandardKernel{T}, x::T, y::T) = pairwise(κ, x, y)
kernel{T}(κ::StandardKernel{T}, x::AbstractArray{T}, y::AbstractArray{T}) = pairwise(κ, x, y)

kernel{T}(κ::KernelComposition{T}, x::T, y::T) = phi(κ.phi, pairwise(κ.kappa, x, y))
function kernel{T}(κ::KernelComposition{T}, x::AbstractArray{T}, y::AbstractArray{T})
    phi(κ.phi, pairwise(κ.kappa, x, y))
end


#==========================================================================
  Generic kernelmatrix Functions
==========================================================================#

function phi_rectangular!{T}(ϕ::CompositionClass{T}, K::AbstractMatrix{T})
    for i in eachindex(K)
        @inbounds K[i] = phi(ϕ, K[i])
    end
    K
end

function phi_symmetric!{T}(ϕ::CompositionClass{T}, K::AbstractMatrix{T})
    if !((n = size(K,1)) == size(K,2))
        throw(DimensionMismatch("Kernel matrix must be square."))
    end
    for j = 1:n, i = (1:j)
        @inbounds K[i,j] = phi(ϕ, K[i,j])
    end
    LinAlg.copytri!(K, 'U')
end

function kernelmatrix{T}(
        σ::Union{Type{Val{:row}},Type{Val{:col}}},
        κ::Kernel{T}, 
        X::AbstractMatrix{T}
    )
    kernelmatrix!(σ, init_pairwisematrix(σ, X), κ, X)
end

function kernelmatrix{T}(
        κ::Kernel{T},
        X::AbstractMatrix{T}
    )
    kernelmatrix(Val{:row}, κ, X)
end

function kernelmatrix{T}(
        σ::Union{Type{Val{:row}},Type{Val{:col}}},
        κ::Kernel{T}, 
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    kernelmatrix!(σ, init_pairwisematrix(σ, X, Y), κ, X, Y)
end

function kernelmatrix{T}(
        κ::Kernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    kernelmatrix(Val{:row}, κ, X, Y)
end


#== Standard Kernel ==#

function kernelmatrix!{T}(
        σ::Union{Type{Val{:row}},Type{Val{:col}}},
        K::Matrix{T},
        κ::StandardKernel{T},
        X::AbstractMatrix{T}
    )
    pairwisematrix!(σ, K, κ, X)
end

function kernelmatrix!{T}(
        σ::Union{Type{Val{:row}},Type{Val{:col}}},
        K::Matrix{T},
        κ::StandardKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(σ, K, κ, X, Y)
end


#== Composition Kernel ==#

function kernelmatrix!{T}(
        σ::Union{Type{Val{:row}},Type{Val{:col}}},
        K::Matrix{T},
        κ::KernelComposition{T},
        X::AbstractMatrix{T}
    )
    pairwisematrix!(σ, K, κ.kappa, X)
    phi_symmetric!(κ.phi, K)
end

function kernelmatrix!{T}(
        σ::Union{Type{Val{:row}},Type{Val{:col}}},
        K::Matrix{T},
        κ::KernelComposition{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(σ, K, κ.kappa, X, Y)
    phi_rectangular!(κ.phi, K)
end


#== Kernel Operation ==#

for (kernel_object, scalar_op, identity, scalar) in (
        (:KernelProduct, :*, :1, :a),
        (:KernelSum,     :+, :0, :c)
    )
    @eval begin
        function kernelmatrix!{T}(
                σ::Union{Type{Val{:row}},Type{Val{:col}}},
                K::Matrix{T},
                κ::$kernel_object{T},
                X::AbstractMatrix{T}
            )
            kernelmatrix!(σ, K, κ.kappa1, X)
            broadcast!($scalar_op, K, kernelmatrix(σ, similar(K), κ.kappa2, X))
            κ.$scalar == $identity ? K : broadcast!($scalar_op, K, κ.$scalar)
        end

        function kernelmatrix!{T}(
                σ::Union{Type{Val{:row}},Type{Val{:col}}},
                K::Matrix{T},
                κ::$kernel_object{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T}
            )
            kernelmatrix!(σ, K, κ.kappa1, X, Y)
            broadcast!($scalar_op, K, kernelmatrix(σ, similar(K), κ.kappa2, X, Y))
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
