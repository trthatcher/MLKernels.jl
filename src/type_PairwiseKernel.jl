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


#== Additive Kernel ==#

doc"SquaredDistanceKernel() = (x-y)ᵀ(x-y)"
immutable SquaredDistanceKernel{T<:AbstractFloat} <: AdditiveKernel{T} end
SquaredDistanceKernel() = SquaredDistanceKernel{Float64}()
@inline phi{T<:AbstractFloat}(κ::SquaredDistanceKernel{T}, x::T, y::T) = (x-y)^2


doc"SineSquaredKernel(p) = Σⱼsin²(p(xⱼ-yⱼ))"
immutable SineSquaredKernel{T<:AbstractFloat} <: AdditiveKernel{T}
    p::Parameter{T}
    SineSquaredKernel(p::Variable{T}) = new(
        Parameter(p, LowerBound(zero(T), :strict))
    )
end
@outer_constructor(SineSquaredKernel, (π,))
@inline phi{T<:AbstractFloat}(κ::SineSquaredKernel{T}, x::T, y::T) = sin(κ.p*(x-y))^2


doc"ChiSquaredKernel() = Σⱼ(xⱼ-yⱼ)²/(xⱼ+yⱼ)"
immutable ChiSquaredKernel{T<:AbstractFloat} <: AdditiveKernel{T} end
ChiSquaredKernel() = ChiSquaredKernel{Float64}()
@inline function phi{T<:AbstractFloat}(κ::ChiSquaredKernel{T}, x::T, y::T)
    (x == y == zero(T)) ? zero(T) : (x-y)^2/(x+y)
end


doc"ScalarProductKernel() = xᵀy"
immutable ScalarProductKernel{T<:AbstractFloat} <: AdditiveKernel{T} end
ScalarProductKernel() = ScalarProductKernel{Float64}()
@inline phi{T<:AbstractFloat}(κ::ScalarProductKernel{T}, x::T, y::T) = x*y


#== Properties of Kernel Classes ==#

for (classobj, properties) in (
        (SquaredDistanceKernel, (false, true,  false, true,  true)),
        (SineSquaredKernel,     (false, true,  false, true,  true)),
        (ChiSquaredKernel,      (false, true,  false, true,  true)),
        (ScalarProductKernel,   (true,  false, true,  true,  true))
    )
    ismercer(::classobj) = properties[1]
    isnegdef(::classobj) = properties[2]
    attainsnegative(::classobj) = properties[3]
    attainszero(::classobj)     = properties[4]
    attainspositive(::classobj) = properties[5]
end
