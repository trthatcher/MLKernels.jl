#===================================================================================================
  Kernel Standardization
===================================================================================================#

function centerkernelmatrix!{T<:AbstractFloat}(
        K::Matrix{T},
        μx_κ::Vector{T},
        μy_κ::Vector{T},
        μ_κ::T = mean(K)
    )
    n, m = size(K)
    length(μx_κ) == n || throw(DimensionMismatch("Kernel statistics do not match matrix."))
    length(μy_κ) == m || throw(DimensionMismatch("Kernel statistics do not match matrix."))
    for j = 1:m, i = 1:n
        @inbounds K[i,j] += μ_κ - μx_κ[i] - μy_κ[j]
    end
    return K
end

function centerkernel!{T<:AbstractFloat}(K::Matrix{T})
    centerkernelmatrix!(K, vec(mean(K,2)), vec(mean(K,1)), mean(K))
end
centerkernel{T<:AbstractFloat}(K::Matrix{T}) = centerkernel!(copy(K))


#== Kernel Centerer ==#

type KernelCenterer{T<:AbstractFloat}
    mux_kappa::Vector{T}
    mu_kappa::T
end
function KernelCenterer{T<:AbstractFloat}(K::Matrix{T})
    μx_κ = vec(mean(K,2))
    μ_κ = mean(μx_κ)
    KernelCenterer{T}(μx_κ, μ_κ)
end

function centerkernel!{T<:AbstractFloat}(KC::KernelCenterer{T}, K::Matrix{T})
    centerkernelmatrix!(K, KC.mux_kappa, vec(mean(K,1)), KC.mu_kappa)
end
centerkernel{T<:AbstractFloat}(KC::KernelCenterer{T}, K::Matrix{T}) = centerkernel!(KC, copy(K))


#== Kernel Transformer ==#

type KernelTransformer{T<:AbstractFloat}
    order::MemoryOrder
    kappa::RealFunction{T}
    X::AbstractMatrix{T}
    KC::KernelCenterer{T}
end

function KernelTransformer{T<:AbstractFloat}(
        σ::MemoryOrder,
        κ::RealFunction{T},
        X::AbstractMatrix{T},
        copy_X::Bool = true
    )
    KC = KernelCenterer(kernelmatrix(σ, κ, X, true))
    KernelTransformer{T}(σ, deepcopy(κ), copy_X ? copy(X) : X, KC)
end

function KernelTransformer{T1<:Real,T2<:Real}(
        σ::MemoryOrder,
        κ::RealFunction{T1},
        X::AbstractMatrix{T2}
    )
    T = promote_type_float(T1, T2)
    KernelTransformer(σ, convert(RealFunction{T}, κ), convert(AbstractMatrix{T}, X), false)
end

function pairwisematrix!{T<:AbstractFloat}(
        K::Matrix{T},
        KT::KernelTransformer{T},
        Y::AbstractMatrix{T}
    )
    K = kernelmatrix!(K, KT.order, K, KT.kappa, KT.X, Y)
    centerkernel!(KT.KC, K)
end
function pairwisematrix{T<:AbstractFloat}(KT::KernelTransformer{T}, Y::AbstractMatrix{T})
    pairwisematrix!(init_pairwise(KT.order, KT.X, Y), KT, Y)
end


#===================================================================================================
  Nystrom Approximation
===================================================================================================#

function srswr(n::Integer, r::AbstractFloat)
    0 < r <= 1 || error("Sample rate must be between 0 and 1")
    ns = Int64(trunc(n*r))
    return Int64[Int64(trunc(u*n +1)) for u in rand(ns)]
end

function nystrom{T<:AbstractFloat}(
        σ::MemoryOrder,
        f::RealFunction{T},
        X::Matrix{T},
        s::Vector{Int64}
    )
    n = length(s)
    Xs = σ == Val{:row} ? X[s,:] : X[:,s]
    C = kernelmatrix(σ, f, Xs, X)
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

function nystrom{T<:AbstractFloat}(
        σ::MemoryOrder,
        f::RealFunction{T},
        X::Matrix{T},
        r::AbstractFloat = 0.15
    )
    n = size(X, σ == Val{:row} ? 1 : 2)
    s = srswr(n, r)
    nystrom(σ, f, X, s)
end

immutable NystromFact{T<:AbstractFloat}
    W::Matrix{T}
    C::Matrix{T}
end

function NystromFact{T<:AbstractFloat}(
        σ::MemoryOrder,
        f::RealFunction{T},
        X::Matrix{T},
        r::AbstractFloat = 0.15
    )
    W, C = nystrom(σ, f, X, r)
    NystromFact{T}(W, C)
end

function NystromFact{T<:AbstractFloat}(
        f::RealFunction{T},
        X::Matrix{T},
        r::AbstractFloat = 0.15
    )
    NystromFact(Val{:row}, f, X, r)
end

function pairwisematrix{T<:AbstractFloat}(CᵀWC::NystromFact{T})
    W = CᵀWC.W
    C = CᵀWC.C
    At_mul_B(C,W)*C
end
