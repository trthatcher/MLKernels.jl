#===================================================================================================
  Nystrom Approximation
===================================================================================================#

for layout in (RowMajor, ColumnMajor)
    (dim, S, fulldim) = layout == RowMajor ? (1, :S, :(:)) : (2, :(:), :S)

    @eval begin
        function samplematrix{T<:AbstractFloat}(
                σ::$layout,
                X::Matrix,
                r::T
            )
            0 < r <= 1 || error("Sample rate must be in range (0,1]")
            n = size(X, $dim)
            s = max(Int64(trunc(n*r)),1)
            S = [rand(1:n) for i = 1:s]
            
        end

        function nystrom_sample{T<:AbstractFloat,U<:Integer}(
                σ::$layout,
                κ::Kernel{T},
                X::Matrix{T},
                S::Vector{U}
            )
            Xs = getindex(X, $S, $fulldim)
            C = kernelmatrix(σ, κ, Xs, X)   # kernel matrix of X and sampled X
            Cs = getindex(C, :, S)  # purely sampled component of C
            return (C, Cs)
        end

    end
end

function nystrom_pinv!{T<:Base.LinAlg.BlasReal}(Cs::Matrix{T})
    # Compute eigendecomposition of sampled component of C
    tol = eps(T)*size(Cs,2)
    QΛQᵀ = eigfact!(Symmetric(Cs))

    # Solve for D = Λ^(-1/2) (pseudo inverse - use tolerance from before factorization)
    D = QΛQᵀ.values
    for i in eachindex(D)
        @inbounds D[i] = abs(D[i]) <= tol ? zero(T) : 1/sqrt(D[i])
    end

    # Scale eigenvectors by D
    Q = QΛQᵀ.vectors
    QD = scale!(Q, D)  # Scales column i of Q by D[i]

    # W := (QD)(QD)ᵀ = (QΛQᵀ)^(-1)  (pseudo inverse)
    W = LinAlg.syrk_wrapper!(similar(QD), 'N', QD)

    return LinAlg.copytri!(W, 'U')
end

immutable NystromFact{T<:Base.LinAlg.BlasReal}
    W::Matrix{T}
    C::Matrix{T}
end

function NystromFact{T<:Base.LinAlg.BlasReal}(
        σ::MemoryLayout,
        κ::Kernel{T},
        X::Matrix{T},
        r::AbstractFloat = convert(T,0.15)
    )
    C, Cs = nystrom_sample(σ, κ, X, samplematrix(σ, X, r))
    W = nystrom_pinv!(Cs)
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
