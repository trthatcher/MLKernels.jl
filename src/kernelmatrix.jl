#================================================
  Generic Kernel Vector Operation
================================================#

function kernel{T}(κ::Kernel{T}, x::T, y::T)
    kappa(κ, pairwise(pairwisefunction(κ), x, y))
end

function kernel{T}(κ::Kernel{T}, x::AbstractArray{T}, y::AbstractArray{T})
    kappa(κ, pairwise(pairwisefunction(κ), x, y))
end



#================================================
  Generic Kernel Matrix Calculation
================================================#

function kappamatrix!{T}(κ::Kernel{T}, P::AbstractMatrix{T})
    for i in eachindex(P)
        @inbounds P[i] = kappa(κ, P[i])
    end
    P
end

function symmetric_kappamatrix!{T}(κ::Kernel{T}, P::AbstractMatrix{T}, symmetrize::Bool)
    if !((n = size(P,1)) == size(P,2))
        throw(DimensionMismatch("Pairwise matrix must be square."))
    end
    for j = 1:n, i = (1:j)
        @inbounds P[i,j] = kappa(κ, P[i,j])
    end
    symmetrize ? LinAlg.copytri!(P, 'U') : P
end

function kernelmatrix!{T}(
        σ::MemoryLayout,
        P::Matrix{T}, 
        κ::Kernel{T},
        X::AbstractMatrix{T},
        symmetrize::Bool
    )
    pairwisematrix!(σ, P, pairwisefunction(κ), X, false)
    symmetric_kappamatrix!(κ, P, symmetrize)
end

function kernelmatrix!{T<:AbstractFloat}(
        σ::MemoryLayout,
        P::Matrix{T}, 
        κ::Kernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(σ, P, pairwisefunction(κ), X, Y)
    kappamatrix!(κ, P)
end

function kernelmatrix{T}(
        σ::MemoryLayout,
        κ::Kernel{T},
        X::AbstractMatrix{T},
        symmetrize::Bool = true
    )
    kernelmatrix!(σ, allocate_pairwisematrix(σ, X), κ, X, symmetrize)
end

function kernelmatrix{T}(
        κ::Kernel{T},
        X::AbstractMatrix{T},
        symmetrize::Bool = true
    )
    σ = RowMajor()
    kernelmatrix!(σ, allocate_pairwisematrix(σ, X), κ, X, symmetrize)
end

function kernelmatrix{T<:AbstractFloat}(
        σ::MemoryLayout,
        κ::Kernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    kernelmatrix!(σ, allocate_pairwisematrix(σ, X, Y), κ, X, Y)
end

function kernelmatrix{T<:AbstractFloat}(
        κ::Kernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    σ = RowMajor()
    kernelmatrix!(σ, allocate_pairwisematrix(σ, X, Y), κ, X, Y)
end
