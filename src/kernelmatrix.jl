



#================================================
  CompositeKernel Matrix Operation
================================================#

@inline function pairwise{T<:AbstractFloat}(h::CompositeKernel{T}, x::T, y::T)
    composition(h.g, pairwise(h.f, x, y))
end

@inline function unsafe_pairwise{T<:AbstractFloat}(
        h::CompositeKernel{T},
        x::AbstractArray{T},
        y::AbstractArray{T}
    )
    composition(h.g, unsafe_pairwise(h.f, x, y))
end

function rectangular_compose!{T<:AbstractFloat}(g::CompositionClass{T}, P::AbstractMatrix{T})
    for i in eachindex(P)
        @inbounds P[i] = composition(g, P[i])
    end
    P
end

function symmetric_compose!{T<:AbstractFloat}(
        g::CompositionClass{T},
        P::AbstractMatrix{T},
        symmetrize::Bool
    )
    if !((n = size(P,1)) == size(P,2))
        throw(DimensionMismatch("PairwiseFunction matrix must be square."))
    end
    for j = 1:n, i = (1:j)
        @inbounds P[i,j] = composition(g, P[i,j])
    end
    symmetrize ? LinAlg.copytri!(P, 'U') : P
end

function pairwisematrix!{T<:AbstractFloat}(
        σ::MemoryOrder,
        P::Matrix{T}, 
        h::CompositeKernel{T},
        X::AbstractMatrix{T},
        symmetrize::Bool
    )
    pairwisematrix!(σ, P, h.f, X, false)
    symmetric_compose!(h.g, P, symmetrize)
end

function pairwisematrix!{T<:AbstractFloat}(
        σ::MemoryOrder,
        P::Matrix{T}, 
        h::CompositeKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(σ, P, h.f, X, Y)
    rectangular_compose!(h.g, P)
end


#================================================
  PointwisePairwiseFunction Matrix Operation
================================================#

@inline pairwise{T<:AbstractFloat}(h::AffineKernel{T}, x::T, y::T) = h.a*pairwise(h.f, x, y) + h.c

@inline function unsafe_pairwise{T<:AbstractFloat}(
        h::AffineKernel{T},
        x::AbstractArray{T},
        y::AbstractArray{T}
    )
    h.a*unsafe_pairwise(h.f, x, y) + h.c
end

function rectangular_affine!{T<:AbstractFloat}(P::AbstractMatrix{T}, a::T, c::T)
    for i in eachindex(P)
        @inbounds P[i] = a*P[i] + c
    end
    P
end

function symmetric_affine!{T<:AbstractFloat}(P::AbstractMatrix{T}, a::T, c::T, symmetrize::Bool)
    if !((n = size(P,1)) == size(P,2))
        throw(DimensionMismatch("symmetric_affine! matrix must be square."))
    end
    for j = 1:n, i = (1:j)
        @inbounds P[i,j] = a*P[i,j] + c
    end
    symmetrize ? LinAlg.copytri!(P, 'U') : P
end

function pairwisematrix!{T<:AbstractFloat}(
        σ::MemoryOrder,
        P::Matrix{T},
        h::AffineKernel{T},
        X::AbstractMatrix{T},
        symmetrize::Bool = true
    )
    pairwisematrix!(σ, P, h.f, X, false)
    symmetric_affine!(P, h.a.value, h.c.value, symmetrize)
end

function pairwisematrix!{T<:AbstractFloat}(
        σ::MemoryOrder,
        P::Matrix{T},
        h::AffineKernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(σ, P, h.f, X, Y)
    rectangular_affine!(P, h.a.value, h.c.value)
end

for (f_obj, scalar_op, identity, scalar) in (
        (:KernelProduct, :*, :1, :a),
        (:KernelSum,     :+, :0, :c)
    )
    @eval begin

        @inline function pairwise{T<:AbstractFloat}(h::$f_obj{T}, x::T, y::T)
            $scalar_op(pairwise(h.f, x, y), pairwise(h.g, x, y), h.$scalar)
        end

        @inline function unsafe_pairwise{T<:AbstractFloat}(
                h::$f_obj{T},
                x::AbstractArray{T},
                y::AbstractArray{T}
            )
            $scalar_op(unsafe_pairwise(h.f, x, y), unsafe_pairwise(h.g, x, y), h.$scalar)
        end

        function pairwisematrix!{T<:AbstractFloat}(
                σ::MemoryOrder,
                P::Matrix{T},
                h::$f_obj{T},
                X::AbstractMatrix{T},
                symmetrize::Bool = true
            )
            pairwisematrix!(σ, P, h.f, X, false)
            broadcast!($scalar_op, P, P, pairwisematrix!(σ, similar(P), h.g, X, false))
            if h.$scalar != $identity
                broadcast!($scalar_op, P, h.$scalar)
            end
            symmetrize ? LinAlg.copytri!(P, 'U') : P
        end

        function pairwisematrix!{T<:AbstractFloat}(
                σ::MemoryOrder,
                P::Matrix{T},
                h::$f_obj{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T}
            )
            pairwisematrix!(σ, P, h.f, X, Y)
            broadcast!($scalar_op, P, P, pairwisematrix!(σ, similar(P), h.g, X, Y))
            h.$scalar == $identity ? P : broadcast!($scalar_op, P, h.$scalar)
        end
    end
end



