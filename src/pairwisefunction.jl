#===================================================================================================
  Pairwise Kernels - Consume two vectors
===================================================================================================#

abstract PairwiseFunction{T<:AbstractFloat} <: RealFunction{T}

pairwise_initiate{T}(::PairwiseFunction{T}) = zero(T)
pairwise_return{T}(::PairwiseFunction{T}, s::T) = s

function description_string(f::PairwiseFunction, showtype::Bool = true)
    obj = typeof(f)
    fields = fieldnames(obj)
    obj_str = string(obj.name.name) * (showtype ? string("{", eltype(f), "}") : "")
    if length(fields) == 0
        return obj_str
    else
        fields_str = join(["$field=$(getfield(f,field).value)" for field in fields], ",")
        *(obj_str, "(", fields_str, ")")
    end
end

function =={T<:PairwiseFunction}(f1::T, f2::T)
    all([getfield(f1,field) == getfield(f2,field) for field in fieldnames(T)])
end


abstract SymmetricPairwiseFunction{T<:AbstractFloat} <: PairwiseFunction{T}


#=== Metrics ===#

abstract Metric{T<:AbstractFloat} <: SymmetricPairwiseFunction{T}

@inline attainsnegative(::Metric) = false


doc"Euclidean() = √((x-y)ᵀ(x-y))"
immutable Euclidean{T<:AbstractFloat} <: Metric{T} end
Euclidean() = Euclidean{Float64}()

convert{T}(::Type{Euclidean{T}}, ::Euclidean) = Euclidean{T}()

isnegdef(::SquaredEuclidean) = true

@inline pairwise_aggregate{T}(::Euclidean{T}, s::T, x::T, y::T) = s + (x-y)^2
@inline pairwise_return{T}(::Euclidean{T}, s::T) = sqrt(s)


doc"SquaredEuclidean() = (x-y)ᵀ(x-y)"
immutable SquaredEuclidean{T<:AbstractFloat} <: Metric{T} end
SquaredEuclidean() = SquaredEuclidean{Float64}()

convert{T}(::Type{SquaredEuclidean{T}}, ::SquaredEuclidean) = SquaredEuclidean{T}()

isnegdef(::SquaredEuclidean) = true

@inline pairwise_aggregate{T}(f::EuclideanDistance{T}, s::T, x::T, y::T) = s + (x-y)^2


doc"ChiSquared() = Σⱼ(xⱼ-yⱼ)²/(xⱼ+yⱼ)"
immutable ChiSquared{T<:AbstractFloat} <: Metric{T} end
ChiSquared() = ChiSquared{Float64}()

convert{T}(::Type{ChiSquared{T}}, ::ChiSquared) = ChiSquared{T}()

isnegdef(::ChiSquared) = true

@inline function pairwise_aggregate{T}(f::ChiSquared{T}, s::T, x::T, y::T)
    s + (x == y == zero(T)) ? zero(T) : (x-y)^2/(x+y)
end


#=== Inner Products ===#

abstract InnerProduct{T<:AbstractFloat} <: SymmetricPairwiseFunction{T}

doc"ScalarProduct() = xᵀy"
immutable ScalarProduct{T<:AbstractFloat} <: InnerProduct{T} end
ScalarProduct() = ScalarProduct{Float64}()

convert{T}(::Type{ScalarProduct{T}}, ::ScalarProduct) = ScalarProduct{T}()

@inline ismercer(::ScalarProduct) = true

@inline pairwise_aggregate{T}(f::ScalarProduct{T}, s::T, x::T, y::T) = s + x*y


#=== Other Functions ===#

doc"SineSquared(p) = Σⱼsin²(p(xⱼ-yⱼ))"
immutable SineSquared{T<:AbstractFloat} <: SymmetricPairwiseFunction{T}
    p::HyperParameter{T}
    SineSquared(p::Variable{T}) = new(
        HyperParameter(p, leftbounded(zero(T), :open))
    )
end
@outer_constructor(SineSquared, (π,))

isnegdef(::SineSquared) = true

@inline pairwise_aggregate{T}(f::SineSquared{T}, s::T, x::T, y::T) = s + sin(f.p*(x-y))^2
