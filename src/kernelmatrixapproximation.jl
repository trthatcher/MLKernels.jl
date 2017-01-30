#===================================================================================================
  Nystrom Approximation
===================================================================================================#

for layout in (RowMajor, ColumnMajor)
    isrowmajor = layout == RowMajor
    @eval begin
        function samplematrix(
                σ::$layout,
                X::Matrix{T},
                r::T
            )
            0 < r <= 1 || error("Sample rate must be in range (0,1]")
            n = size(X,$(isrowmajor ? 1 : 2))
            s = max(Int64(trunc(n*r)),1)
            S = [rand(1:n) for i = 1:s]
            X = getindex(X, $(isrowmajor ? :S : :(:)), $(isrowmajor ? :(:) : :S))
        end
    end
end

function nystrom{T<:Base.LinAlg.BlasReal}(
        σ::MemoryOrder,
        κ::Kernel{T},
        X::Matrix{T},
        Xs::Matrix{T}
    )
    # Get kernel matrix of X and Xs (sampled observations of X)
    C = kernelmatrix(σ, κ, Xs, X)

    # Compute eigendecomposition and get D
    tol = eps(T)*n
    QΛQᵀ = eigfact!(Symmetric(C[:,s]))

    # Solve for D = Λ^(-1/2) (pseudo inverse - use tolerance from before factorization)
    D = QΛQᵀ.values
    for i in eachindex(D)
        @inbounds D[i] = abs(D[i]) <= tol ? zero(T) : 1/sqrt(D[i])
    end

    # Scale eigenvectors by D
    V = VΛVᵀ.vectors
    VD = scale!(V, D)  # Scales column i of V by D[i]

    # W := (VD)(VD)ᵀ = (VΛVᵀ)^(-1)  (pseudo inverse)
    W = LinAlg.syrk_wrapper!(Array(T,n,n), 'N', DV)

    return (LinAlg.copytri!(W, 'U'), C)
end

immutable NystromFact{T<:Base.LinAlg.BlasReal}
    W::Matrix{T}
    C::Matrix{T}
end

function NystromFact{T<:Base.LinAlg.BlasReal}(
        σ::MemoryOrder,
        κ::Kernel{T},
        X::Matrix{T},
        r::AbstractFloat = convert(T,0.15)
    )
    W, C = nystrom(σ, κ, X, samplematrix(X,r))
    NystromFact{T}(W, C)
end

function NystromFact{T<:Base.LinAlg.BlasReal}(
        κ::Kernel{T},
        X::Matrix{T},
        r::AbstractFloat = 0.15
    )
    NystromFact(RowMajor(), κ, X, r)
end

function pairwisematrix{T<:Base.LinAlg.BlasReal}(CᵀWC::NystromFact{T})
    W = CᵀWC.W
    C = CᵀWC.C
    At_mul_B(C,W)*C
end
