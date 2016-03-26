#===================================================================================================
  Pairwise Kernels - Consume two vectors
===================================================================================================#

abstract PairwiseKernel{T<:AbstractFloat} <: StandardKernel{T}
abstract AdditiveKernel{T<:AbstractFloat} <: PairwiseKernel{T}

function description_string(κ::PairwiseKernel)
    obj = typeof(κ)
    fields = fieldnames(obj)
    obj_str = string(obj.name.name)
    if length(fields) == 0
        return obj_str
    else
        fields_str = join(["$field=$(getfield(κ,field).value)" for field in fields], ",")
        *(obj_str, "(", fields_str, ")")
    end
end


#== Non-Negative Negative Definite Additive Kernel ==#

abstract NonNegNegDefAdditiveKernel{T<:AbstractFloat} <: AdditiveKernel{T}
@inline isnegdef(::NonNegNegDefAdditiveKernel)        = true
@inline attainsnegative(::NonNegNegDefAdditiveKernel) = false

doc"SquaredDistanceKernel() = (x-y)ᵀ(x-y)"
immutable SquaredDistanceKernel{T<:AbstractFloat} <: NonNegNegDefAdditiveKernel{T} end
SquaredDistanceKernel() = SquaredDistanceKernel{Float64}()
@inline phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T}, x::T, y::T) = (x-y)^2


doc"SineSquaredKernel(p) = Σⱼsin²(p(xⱼ-yⱼ))"
immutable SineSquaredKernel{T<:AbstractFloat} <: NonNegNegDefAdditiveKernel{T}
    p::Parameter{T}
    SineSquaredKernel(p::Variable{T}) = new(
        Parameter(p, LowerBound(zero(T), :strict))
    )
end
@outer_constructor(SineSquaredKernel, (π,))
@inline phi{T<:AbstractFloat}(κ::SineSquaredKernel{T}, x::T, y::T) = sin(κ.p*(x-y))^2


doc"ChiSquaredKernel() = Σⱼ(xⱼ-yⱼ)²/(xⱼ+yⱼ)"
immutable ChiSquaredKernel{T<:AbstractFloat} <: NonNegNegDefAdditiveKernel{T} end
ChiSquaredKernel() = ChiSquaredKernel{Float64}()
@inline function phi{T<:AbstractFloat}(κ::ChiSquaredKernel{T}, x::T, y::T)
    (x == y == zero(T)) ? zero(T) : (x-y)^2/(x+y)
end


#== Non-Negative Negative Definite Additive Kernel ==#

doc"ScalarProductKernel() = xᵀy"
immutable ScalarProductKernel{T<:AbstractFloat} <: AdditiveKernel{T} end
ScalarProductKernel() = ScalarProductKernel{Float64}()
@inline phi{T<:AbstractFloat}(κ::ScalarProductKernel{T}, x::T, y::T) = x*y
@inline ismercer(::ScalarProductKernel) = true
