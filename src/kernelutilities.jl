#=
type KernelStandardizer{T<:AbstractFloat}
    normalize::Bool
    center::Bool
    kappa::Vector{T}
    k::T
end

KernelStandardizer{T<:AbstractFloat}(X::Matrix{T}

centerkernelmatrix!(K::AbstractMatrix{T}, diag_K::Vector{T}, 
=#

#=
function rankedapproximation!{T<:AbstractFloat}(W::Matrix{T}, D::Vector{T}, U::Matrix{T}, k::Integer)
    n = size(W,1)
    for j = 1:n, i = 1:j
        s = zero(T)
        @simd for l = 1:k
            @inbounds Uik = U[i,k]
            @inbounds Ujk = U[j,k]
            tmp = Uik * Ujk
            s += D[k]*tmp
        end
        W[i,j] = s
    end
    W
end
=#

#eps(real(float(one(eltype(M)))))*maximum(size(A))
function nystrom{T<:AbstractFloat,U<:Integer}(
        σ::MemoryOrder,
        κ::RealFunction{T},
        X::Matrix{T},
        s::Vector{U}
    )
    n = length(s)
    C = kernelmatrix(σ, κ, σ == Val{:row} ? X[s,:] : X[:,s], X)
    tol = eps(T)*n
    VDVᵀ = eigfact!(Symmetric(C[:,s]))
    D = VDVᵀ.values
    for i in eachindex(D)
        @inbounds D[i] = abs(D[i]) <= tol ? zero(T) : 1/sqrt(D[i])
    end
    V = VDVᵀ.vectors
    DV = scale!(V,D)
    W = LinAlg.syrk_wrapper!(Array(T,n,n), 'N', DV)
    return (LinAlg.copytri!(W, 'U'), C)
end

immutable NystromFact{T<:AbstractFloat}
    W::Matrix{T}
    C::Matrix{T}
end

function srswr(n::Integer, r::AbstractFloat)
    0 <= r <= 1 || error("Sample rate must be between 0 and 1")
    convert(Vector{Int64}, trunc(rand(Int64(trunc(n*r)))*n+1))
end

function NystromFact{T<:AbstractFloat,U<:Integer}(
        σ::MemoryOrder,
        κ::RealFunction{T},
        X::Matrix{T},
        s::Vector{U} = srswr(size(X, σ == Val{:row} ? 1 : 2), 0.15)
    )
    NystromFact{T}(nystrom(σ, κ, X, s)...)
end

function pairwisematrix{T<:AbstractFloat}(CᵀWC::NystromFact{T})
    W = CᵀWC.W
    C = CᵀWC.C
    At_mul_B(C,W)*C
end
