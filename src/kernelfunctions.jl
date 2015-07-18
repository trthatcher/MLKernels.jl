#==========================================================================
  Kernel Functions
==========================================================================#

kernel{T<:FloatingPoint}(κ::BaseKernel{T}, x::T, y::T) = pairwise(κ, x, y)
kernel{T<:FloatingPoint}(κ::BaseKernel{T}, x::Vector{T}, y::Vector{T}) = pairwise(κ, x, y)

kernel{T<:FloatingPoint}(κ::ARD{T}, x::T, y::T) = pairwise(κ.k, x, y, κ.w[1])
kernel{T<:FloatingPoint}(κ::ARD{T}, x::Vector{T}, y::Vector{T}) = pairwise(κ.k, x, y, κ.w)

kernel{T<:FloatingPoint}(κ::CompositeKernel{T}, x::T, y::T) = kappa(κ, pairwise(κ.k, x, y))
kernel{T<:FloatingPoint}(κ::CompositeKernel{T}, x::Vector{T}, y::Vector{T}) = kappa(κ, pairwise(κ.k, x, y))

kernel{T<:FloatingPoint}(ψ::KernelSum{T}, x::T, y::T) = sum(ψ.a, sym(map(κ -> kernel(κ,x,y), ψ.k)))
kernel{T<:FloatingPoint}(ψ::KernelSum{T}, x::Vector{T}, y::Vector{T}) = sum(ψ.a, sym(map(κ -> kernel(κ,x,y), ψ.k)))

kernel{T<:FloatingPoint}(ψ::KernelProduct{T}, x::T, y::T) = sum(ψ.a, sym(map(κ -> kernel(κ,x,y), ψ.k)))
kernel{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T}) = prod(ψ.a, prod(map(κ -> kernel(κ,x,y), ψ.k)))


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

function kernelmatrix{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', symmetrize::Bool = true)
    is_trans = trans == 'T'
    kernelmatrix!(init_pairwise(X, is_trans), κ, X, is_trans, uplo == 'U', symmetrize)
end

function kernelmatrix{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    is_trans = trans == 'T'
    kernelmatrix!(init_pairwise(X, Y, is_trans), κ, X, Y, is_trans)
end


#==========================================================================
  Base Kernel Matrix Functions
==========================================================================#

function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::BaseKernel{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool, symmetrize::Bool)
    pairwise!(K, κ, X, is_trans, store_upper)
    symmetrize ? (store_upper ? syml!(K) : symu!(K)) : K
end

function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::BaseKernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
    pairwise!(K, κ, X, Y, is_trans)
end


#==========================================================================
  Composite Kernel Matrix Functions
==========================================================================#


function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::CompositeKernel{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool, symmetrize::Bool)
    pairwise!(K, κ.k, X, is_trans, store_upper)
    kappa_square_matrix!(κ, K, store_upper)
    symmetrize ? (store_upper ? syml!(K) : symu!(K)) : K
end

function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::CompositeKernel{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
    pairwise!(K, κ.k, X, Y, is_trans)
    kappa_matrix!(κ, K)
end


#==========================================================================
  Combination Kernel Matrix Functions
==========================================================================#

for (kernel_object, matrix_op, array_op, identity) in (
        (:KernelProduct, :matrix_prod!, :scale!,     :1),
        (:KernelSum,     :matrix_sum!,  :translate!, :0)
    )
    @eval begin

        function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::$kernel_object{T}, X::Matrix{T}, is_trans::Bool, store_upper::Bool, symmetrize::Bool)
            c = length(κ.k)
            kernelmatrix!(K, κ.k[1], X, is_trans, store_upper, false)
            if c > 1
                K_factor = similar(K)
                for i = 2:c
                    ($matrix_op)(K, kernelmatrix!(K_factor, κ.k[i], X, is_trans, store_upper, false), store_upper, false)
                end
            end
            K = symmetrize ? (store_upper ? syml!(K) : symu!(K)) : K
            κ.a == $identity ? K : ($array_op)(κ.a, K)
        end

        function kernelmatrix!{T<:FloatingPoint}(K::Matrix{T}, κ::$kernel_object{T}, X::Matrix{T}, Y::Matrix{T}, is_trans::Bool)
            c = length(κ.k)
            kernelmatrix!(K, κ.k[1], X, Y, is_trans)
            if c > 1
                K_factor = similar(K)
                for i = 2:c
                    ($matrix_op)(K, kernelmatrix!(K_factor, κ.k[i], X, Y, is_trans))
                end
            end
            κ.a == $identity ? K : ($array_op)(κ.a, K)
        end

    end
end
