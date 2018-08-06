#================================================
  Generic Kernel Vector Operation
================================================#

function kernel(κ::Kernel{T}, x::T, y::T) where {T<:AbstractFloat}
    kappa(κ, pairwise(pairwisefunction(κ), x, y))
end

function kernel(κ::Kernel{T}, x::AbstractArray{T}, y::AbstractArray{T}) where {T<:AbstractFloat}
    kappa(κ, pairwise(pairwisefunction(κ), x, y))
end



#================================================
  Generic Kernel Matrix Calculation
================================================#

function kappamatrix!(κ::Kernel{T}, P::AbstractMatrix{T}) where {T<:AbstractFloat}
    for i in eachindex(P)
        @inbounds P[i] = kappa(κ, P[i])
    end
    P
end

function symmetric_kappamatrix!(
        κ::Kernel{T},
        P::AbstractMatrix{T},
        symmetrize::Bool
    ) where {T<:AbstractFloat}
    if !((n = size(P,1)) == size(P,2))
        throw(DimensionMismatch("Pairwise matrix must be square."))
    end
    for j = 1:n, i = (1:j)
        @inbounds P[i,j] = kappa(κ, P[i,j])
    end
    symmetrize ? LinearAlgebra.copytri!(P, 'U') : P
end

function kernelmatrix!(
        σ::MemoryLayout,
        P::Matrix{T},
        κ::Kernel{T},
        X::AbstractMatrix{T},
        symmetrize::Bool
    ) where {T<:AbstractFloat}
    pairwisematrix!(σ, P, pairwisefunction(κ), X, false)
    symmetric_kappamatrix!(κ, P, symmetrize)
end

function kernelmatrix!(
        σ::MemoryLayout,
        P::Matrix{T},
        κ::Kernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    ) where {T<:AbstractFloat}
    pairwisematrix!(σ, P, pairwisefunction(κ), X, Y)
    kappamatrix!(κ, P)
end

function kernelmatrix(
        σ::MemoryLayout,
        κ::Kernel{T},
        X::AbstractMatrix{T},
        symmetrize::Bool = true
    ) where {T<:AbstractFloat}
    symmetric_kappamatrix!(κ, pairwisematrix(σ, pairwisefunction(κ), X, false), symmetrize)
end

function kernelmatrix(
        σ::MemoryLayout,
        κ::Kernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    ) where {T<:AbstractFloat}
    kappamatrix!(κ, pairwisematrix(σ, pairwisefunction(κ), X, Y))
end



#================================================
  Generic Catch-All Methods
================================================#

function kernel(κ::Kernel{T}, x::Real, y::Real) where {T}
    kernel(κ, T(x), T(y))
end

function kernel(
        κ::Kernel{T},
        x::AbstractArray{T1},
        y::AbstractArray{T2}
    ) where {T,T1,T2}
    kernel(κ, convert(AbstractArray{T}, x), convert(AbstractArray{T}, y))
end

function kernelmatrix(
        σ::MemoryLayout,
        κ::Kernel{T},
        X::AbstractMatrix{T1},
        symmetrize::Bool = true
    ) where {T,T1}
    U = convert(AbstractMatrix{T}, X)
    kernelmatrix(σ, κ, U, symmetrize)
end

function kernelmatrix(
        κ::Kernel,
        X::AbstractMatrix,
        symmetrize::Bool = true
    )
    kernelmatrix(RowMajor(), κ, X, symmetrize)
end

function kernelmatrix(
        σ::MemoryLayout,
        κ::Kernel{T},
        X::AbstractMatrix{T1},
        Y::AbstractMatrix{T2}
    ) where {T,T1,T2}
    U = convert(AbstractMatrix{T}, X)
    V = convert(AbstractMatrix{T}, Y)
    kernelmatrix(σ, κ, U, V)
end

function kernelmatrix(
        κ::Kernel,
        X::AbstractMatrix,
        Y::AbstractMatrix
    )
    kernelmatrix(RowMajor(), κ, X, Y)
end



#===================================================================================================
  Kernel Centering
===================================================================================================#

function centerkernelmatrix!(K::Matrix{T}) where {T<:AbstractFloat}
    μx = Statistics.mean(K, dims = 2)
    μy = Statistics.mean(K, dims = 1)
    μ  = Statistics.mean(K)

    K .+= μ .- μx .- μy

    return K
end
centerkernelmatrix(K::Matrix) = centerkernelmatrix!(copy(K))
