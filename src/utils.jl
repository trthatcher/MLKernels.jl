# Check Arguments Macro ====================================================================

macro check_args(K, param, cond, desc=string(cond))
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(K)), ": ", $(string(param)), " = ", $(esc(param)), " does not ",
                "satisfy the constraint ", $(string(desc)), ".")))
        end
    end
end

#macro default_kernel_convert(K)
#    θ = fieldnames(eval(K))
#    K_name = esc(K)
#    conversion_call = Expr(:call, :($K_name{T}), [:(κ.$θₖ) for θₖ in θ]...)
#    return quote
#        function test(::Type{K}, κ::$K_name) where {K>:$K_name{T}} where T
#            return $conversion_call
#        end
#    end
#end


# Type Rules ===============================================================================

function promote_float(Tₖ::DataType...)
    if length(Tₖ) == 0
        return Float64
    end
    T = promote_type(Tₖ...)
    return T <: AbstractFloat ? T : Float64
end


# Common Functions =========================================================================

for orientation in (:row, :col)

    row_oriented = orientation == :row
    dim_obs, dim_param = row_oriented ? (1, 2) : (2, 1)
    NT, TN = row_oriented ? ('N', 'T') : ('T', 'N')

    @eval begin

        function dotvectors!(
                ::Val{$(Meta.quot(orientation))},
                xᵀx::Vector{T},
                X::Matrix{T}
            ) where {T<:AbstractFloat}
            if !(size(X, $dim_obs) == length(xᵀx))
                errorstring = string("Dimension mismatch on dimension ", $dim_obs)
                throw(DimensionMismatch(errorstring))
            end
            fill!(xᵀx, zero(T))
            for I in Base.Cartesian.CartesianIndices(size(X))
                xᵀx[I.I[$dim_obs]] += X[I]^2
            end
            xᵀx
        end

        @inline function dotvectors(σ::Val{$(Meta.quot(orientation))}, X::Matrix{T}) where {T<:AbstractFloat}
            dotvectors!(σ, Array{T}(undef, size(X,$dim_obs)), X)
        end

        function gramian!(
                ::Val{$(Meta.quot(orientation))},
                G::Matrix{T},
                X::Matrix{T},
                symmetrize::Bool
            ) where {T<:LinearAlgebra.BLAS.BlasReal}
            LinearAlgebra.BLAS.syrk!('U', $NT, one(T), X, zero(T), G)
            symmetrize ? LinearAlgebra.copytri!(G, 'U') : G
        end

        @inline function gramian!(
                ::Val{$(Meta.quot(orientation))},
                G::Matrix{T},
                X::Matrix{T},
                Y::Matrix{T}
            ) where {T<:LinearAlgebra.BLAS.BlasReal}
            LinearAlgebra.BLAS.gemm!($NT, $TN, one(T), X, Y, zero(T), G)
        end
    end
end
