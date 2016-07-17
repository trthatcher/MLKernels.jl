#===================================================================================================
  Generic pairwisematrix functions for kernels consuming two vectors
===================================================================================================#

# Row major and column major ordering are supported
MemoryOrder = Union{Type{Val{:col}},Type{Val{:row}}}

call{T}(f::RealFunction{T}, x::T, y::T) = pairwise(f, x, y)
call{T}(f::RealFunction{T}, x::AbstractArray{T}, y::AbstractArray{T}) = pairwise(f, x, y)


#================================================
  Generic Pairwise Matrix Operation
================================================#

function pairwise{T<:AbstractFloat}(f::RealFunction{T}, x::AbstractArray{T}, y::AbstractArray{T})
    if (n = length(x)) != length(y)
        throw(DimensionMismatch("Arrays x and y must have the same length."))
    end
    n == 0 ? zero(T) : unsafe_pairwise(f, x, y)
end

for (order, dimension) in ((:(:row), 1), (:(:col), 2))
    isrowmajor = order == :(:row)
    @eval begin

        @inline function subvector(::Type{Val{$order}}, X::AbstractMatrix,  i::Integer)
            $(isrowmajor ? :(slice(X, i, :)) : :(slice(X, :, i)))
        end

        @inline function init_pairwisematrix{T<:AbstractFloat}(
                 ::Type{Val{$order}},
                X::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(X,$dimension))
        end

        @inline function init_pairwisematrix{T<:AbstractFloat}(
                 ::Type{Val{$order}},
                X::AbstractMatrix{T}, 
                Y::AbstractMatrix{T}
            )
            Array(T, size(X,$dimension), size(Y,$dimension))
        end

        function checkpairwisedimensions{T<:AbstractFloat}(
                 ::Type{Val{$order}},
                P::Matrix{T}, 
                X::AbstractMatrix{T}
            )
            n = size(P,1)
            if size(P,2) != n
                throw(DimensionMismatch("Pernel matrix P must be square"))
            elseif size(X, $dimension) != n
                errorstring = string("Dimensions of P must match dimension ", $dimension, "of X")
                throw(DimensionMismatch(errorstring))
            end
            return n
        end

        function checkpairwisedimensions(
                 ::Type{Val{$order}},
                P::Matrix,
                X::AbstractMatrix, 
                Y::AbstractMatrix
            )
            n = size(X, $dimension)
            m = size(Y, $dimension)
            if n != size(P,1)
                errorstring = string("Dimension 1 of P must match dimension ", $dimension, "of X")
                throw(DimensionMismatch(errorstring))
            elseif m != size(P,2)
                errorstring = string("Dimension 2 of P must match dimension ", $dimension, "of Y")
                throw(DimensionMismatch(errorstring))
            end
            return (n, m)
        end

        function pairwisematrix!{T<:AbstractFloat}(
                σ::Type{Val{$order}},
                P::Matrix{T}, 
                f::PairwiseFunction{T},
                X::AbstractMatrix{T},
                symmetrize::Bool
            )
            n = checkpairwisedimensions(σ, P, X)
            for j = 1:n
                xj = subvector(σ, X, j)
                for i = 1:j
                    xi = subvector(σ, X, i)
                    @inbounds P[i,j] = unsafe_pairwise(f, xi, xj)
                end
            end
            symmetrize ? LinAlg.copytri!(P, 'U', false) : P
        end

        function pairwisematrix!{T<:AbstractFloat}(
                σ::Type{Val{$order}},
                P::Matrix{T}, 
                f::PairwiseFunction{T},
                X::AbstractMatrix{T},
                Y::AbstractMatrix{T},
            )
            n, m = checkpairwisedimensions(σ, P, X, Y)
            for j = 1:m
                yj = subvector(σ, Y, j)
                for i = 1:n
                    xi = subvector(σ, X, i)
                    @inbounds P[i,j] = unsafe_pairwise(f, xi, yj)
                end
            end
            P
        end
    end
end

function pairwisematrix{T<:AbstractFloat}(
        σ::MemoryOrder,
        f::RealFunction{T}, 
        X::AbstractMatrix{T},
        symmetrize::Bool = true
    )
    pairwisematrix!(σ, init_pairwisematrix(σ, X), f, X, symmetrize)
end

function pairwisematrix(
        f::RealFunction,
        X::AbstractMatrix,
        symmetrize::Bool = true
    )
    pairwisematrix(Val{:row}, f, X, symmetrize)
end

function pairwisematrix{T<:AbstractFloat}(
        σ::MemoryOrder,
        f::RealFunction{T}, 
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(σ, init_pairwisematrix(σ, X, Y), f, X, Y)
end

function pairwisematrix(
        f::RealFunction,
        X::AbstractMatrix,
        Y::AbstractMatrix
    )
    pairwisematrix(Val{:row}, f, X, Y)
end

# Identical definitions
kernel = pairwise
kernelmatrix  = pairwisematrix
kernelmatrix! = pairwisematrix!


#================================================
  PairwiseFunction Scalar/Vector Operation
================================================#

function pairwise{T<:AbstractFloat}(f::PairwiseFunction{T}, x::T, y::T)
    pairwise_return(f, pairwise_aggregate(f, pairwise_initiate(f), x, y))
end

# No checks, assumes length(x) == length(y) >= 1
function unsafe_pairwise{T<:AbstractFloat}(
        f::PairwiseFunction{T},
        x::AbstractArray{T}, 
        y::AbstractArray{T}
    )
    s = pairwise_initiate(f)
    @simd for I in eachindex(x,y)
        @inbounds xi = x[I]
        @inbounds yi = y[I]
        s = pairwise_aggregate(f, s, xi, yi)
    end
    pairwise_return(f, s)
end


#================================================
  CompositeFunction Matrix Operation
================================================#

@inline function pairwise{T<:AbstractFloat}(h::CompositeFunction{T}, x::T, y::T)
    composition(h.g, pairwise(h.f, x, y))
end

@inline function unsafe_pairwise{T<:AbstractFloat}(
        h::CompositeFunction{T},
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
        h::CompositeFunction{T},
        X::AbstractMatrix{T},
        symmetrize::Bool
    )
    pairwisematrix!(σ, P, h.f, X, false)
    symmetric_compose!(h.g, P, symmetrize)
end

function pairwisematrix!{T<:AbstractFloat}(
        σ::MemoryOrder,
        P::Matrix{T}, 
        h::CompositeFunction{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(σ, P, h.f, X, Y)
    rectangular_compose!(h.g, P)
end


#================================================
  PointwiseRealFunction Matrix Operation
================================================#

@inline pairwise{T<:AbstractFloat}(h::AffineFunction{T}, x::T, y::T) = h.a*pairwise(h.f, x, y) + h.c

@inline function unsafe_pairwise{T<:AbstractFloat}(
        h::AffineFunction{T},
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
        h::AffineFunction{T},
        X::AbstractMatrix{T},
        symmetrize::Bool = true
    )
    pairwisematrix!(σ, P, h.f, X, false)
    symmetric_affine!(P, h.a.value, h.c.value, symmetrize)
end

function pairwisematrix!{T<:AbstractFloat}(
        σ::MemoryOrder,
        P::Matrix{T},
        h::AffineFunction{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T}
    )
    pairwisematrix!(σ, P, h.f, X, Y)
    rectangular_affine!(P, h.a.value, h.c.value)
end

for (f_obj, scalar_op, identity, scalar) in (
        (:FunctionProduct, :*, :1, :a),
        (:FunctionSum,     :+, :0, :c)
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


#================================================
  Generic Catch-All Methods
================================================#

function pairwise{T1<:AbstractFloat,T2<:Real,T3<:Real}(f::RealFunction{T1}, x::T2, y::T3)
    T = promote_type(T1, T2, T3)
    pairwise(convert(RealFunction{T}, f), convert(T, x), convert(T, y))
end

function pairwise{T1<:AbstractFloat,T2<:Real,T3<:Real}(
        f::RealFunction{T1},
        x::AbstractArray{T2},
        y::AbstractArray{T3}
    )
    T = promote_type(T1, T2, T3)
    u = convert(AbstractArray{T}, x)
    v = convert(AbstractArray{T}, y)
    pairwise(convert(RealFunction{T}, f), u, v)
end

function pairwisematrix{T1<:AbstractFloat,T2<:Real}(
        σ::MemoryOrder,
        f::RealFunction{T1}, 
        X::AbstractMatrix{T2},
        symmetrize::Bool = true
    )
    T = promote_type(T1, T2)
    U = convert(AbstractMatrix{T}, X)
    pairwisematrix!(σ, init_pairwisematrix(σ, U), convert(RealFunction{T}, f), U, symmetrize)
end

function pairwisematrix{T1<:AbstractFloat,T2<:Real,T3<:Real}(
        σ::MemoryOrder,
        f::RealFunction{T1}, 
        X::AbstractMatrix{T2},
        Y::AbstractMatrix{T3}
    )
    T = promote_type(T1, T2, T3)
    U = convert(AbstractMatrix{T}, X)
    V = convert(AbstractMatrix{T}, Y)
    pairwisematrix!(σ, init_pairwisematrix(σ, U, V), convert(RealFunction{T}, f), U, V)
end





#===================================================================================================
  ScalarProduct and SquaredDistance using BLAS/Built-In methods
===================================================================================================#

for (order, dimension) in ((:(:row), 1), (:(:col), 2))
    isrowmajor = order == :(:row)
    @eval begin

        @inline function pairwisematrix!{T<:AbstractFloat}(
                σ::Type{Val{$order}},
                P::Matrix{T}, 
                f::ScalarProduct{T},
                X::Matrix{T},
                symmetrize::Bool
            )
            gramian!(σ, P, X, symmetrize)
        end

        @inline function pairwisematrix!{T<:AbstractFloat}(
                σ::Type{Val{$order}},
                P::Matrix{T}, 
                f::ScalarProduct{T},
                X::Matrix{T},
                Y::Matrix{T},
            )
            gramian!(σ, P, X, Y)
        end

        function pairwisematrix!{T<:AbstractFloat}(
                σ::Type{Val{$order}},
                P::Matrix{T}, 
                f::SquaredEuclidean{T},
                X::Matrix{T},
                symmetrize::Bool
            )
            gramian!(σ, P, X, false)
            xᵀx = dotvectors(σ, X)
            squared_distance!(P, xᵀx, symmetrize)
        end

        function pairwisematrix!{T<:AbstractFloat}(
                σ::Type{Val{$order}},
                P::Matrix{T}, 
                f::SquaredEuclidean{T},
                X::Matrix{T},
                Y::Matrix{T},
            )
            gramian!(σ, P, X, Y)
            xᵀx = dotvectors(σ, X)
            yᵀy = dotvectors(σ, Y)
            squared_distance!(P, xᵀx, yᵀy)
        end
    end
end

#===================================================================================================
  Generic Catch-Alls
===================================================================================================#
