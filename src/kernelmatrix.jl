#================================================
  Generic Kernel Vector Operation
================================================#

function kernel{T<:AbstractFloat}(κ::Kernel{T}, x::T, y::T)
    kappa(κ, pairwise(pairwisefunction(κ), x, y))
end

function kernel{T<:AbstractFloat}(κ::Kernel{T}, x::AbstractArray{T}, y::AbstractArray{T})
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

function kernelmatrix!{T}(
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
        σ::MemoryLayout,
        κ::Kernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    kernelmatrix!(σ, allocate_pairwisematrix(σ, X, Y), κ, X, Y)
end



#================================================
  Generic Catch-All Methods
================================================#

function kernel{T}(κ::Kernel{T}, x::Real, y::Real)
    kernel(κ, T(x), T(y))
end

function kernel{T,T1,T2}(
        κ::Kernel{T},
        x::AbstractArray{T1},
        y::AbstractArray{T2}
    )
    kernel(κ, convert(AbstractArray{T}, x), convert(AbstractArray{T}, y))
end

function kernelmatrix{T,T1}(
        σ::MemoryLayout,
        κ::Kernel{T}, 
        X::AbstractMatrix{T1},
        symmetrize::Bool = true
    )
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

function kernelmatrix{T,T1,T2}(
        σ::MemoryLayout,
        κ::Kernel{T}, 
        X::AbstractMatrix{T1},
        Y::AbstractMatrix{T2}
    )
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
