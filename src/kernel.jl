#===================================================================================================
  Kernels
===================================================================================================#

abstract type Kernel{T<:AbstractFloat} end

function string(κ::Kernel)
    args = [string(getvalue(getfield(κ,θ))) for θ in fieldnames(typeof(κ))]
    kernelname = typeof(κ).name.name
    string(kernelname, "(", join(args, ","), ")")
end

function show(io::IO, κ::Kernel)
    print(io, string(κ))
end

function pairwisefunction(::Kernel)
    error("No pairwise function specified for kernel")
end

@inline eltype(::Type{<:Kernel{E}}) where {E} = E
@inline eltype(κ::Kernel) = eltype(typeof(κ))

ismercer(::Kernel) = false
isnegdef(::Kernel) = false
isstationary(κ::Kernel) = isstationary(pairwisefunction(κ))
isisotropic(κ::Kernel)  = isisotropic(pairwisefunction(κ))

thetafieldnames(κ::Kernel) = fieldnames(typeof(κ))

gettheta(κ::Kernel{T}) where {T} = T[gettheta(getfield(κ,θ)) for θ in thetafieldnames(κ)]

function settheta!(κ::Kernel{T},v::Vector{T}) where {T}
    fields = thetafieldnames(κ)
    if length(fields) != length(v)
        throw(DimensionMismatch("Update vector has invalid length"))
    end
    for i in eachindex(fields)
        settheta!(getfield(κ, fields[i]), v[i])
    end
    return κ
end

function checktheta(κ::Kernel{T},v::Vector{T}) where {T}
    fields = thetafieldnames(κ)
    if length(fields) != length(v)
        throw(DimensionMismatch("Update vector has invalid length"))
    end
    for i in eachindex(fields)
        if !checktheta(getfield(κ, fields[i]), v[i])
            return false
        end
    end
    return true
end

function floattype(T_i::DataType...)
    T_max = promote_type(T_i...)
    T_max <: AbstractFloat ? T_max : Float64
end


#================================================
  Not True Kernels
================================================#

"SigmoidKernel(a,c) = tanh(a⋅xᵀy + c)   a ∈ (0,∞), c ∈ (0,∞)"
struct SigmoidKernel{T<:AbstractFloat} <: Kernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    SigmoidKernel{T}(a::Real, c::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,a), interval(OpenBound(zero(T)),   nothing)),
        HyperParameter(convert(T,c), interval(ClosedBound(zero(T)), nothing))
    )
end
function SigmoidKernel(a::T1 = 1.0, c::T2 = one(T1)) where {T1<:Real,T2<:Real}
    SigmoidKernel{floattype(T1,T2)}(a,c)
end

@inline sigmoidkernel(z::T, a::T, c::T) where {T<:AbstractFloat} = tanh(a*z + c)

@inline pairwisefunction(::SigmoidKernel) = ScalarProduct()
@inline kappa(κ::SigmoidKernel{T}, z::T) where {T} = sigmoidkernel(z, getvalue(κ.a), getvalue(κ.c))



#================================================
  Mercer Kernels
================================================#

abstract type MercerKernel{T<:AbstractFloat} <: Kernel{T} end
@inline ismercer(::MercerKernel) = true

"ExponentialKernel(α) = exp(-α⋅‖x-y‖)   α ∈ (0,∞)"
struct ExponentialKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    ExponentialKernel{T}(α::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
ExponentialKernel(α::T=1.0) where {T<:Real} = ExponentialKernel{floattype(T)}(α)
LaplacianKernel = ExponentialKernel

@inline exponentialkernel(z::T, α::T) where {T<:AbstractFloat} = exp(-α*sqrt(z))

@inline pairwisefunction(::ExponentialKernel) = SquaredEuclidean()
@inline function kappa(κ::ExponentialKernel{T}, z::T) where {T<:AbstractFloat}
    exponentialkernel(z, getvalue(κ.alpha))
end



"SquaredExponentialKernel(α) = exp(-α⋅‖x-y‖²)   α ∈ (0,∞)"
struct SquaredExponentialKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    SquaredExponentialKernel{T}(α::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
SquaredExponentialKernel(α::T=1.0) where {T<:Real} = SquaredExponentialKernel{floattype(T)}(α)
GaussianKernel = SquaredExponentialKernel
RadialBasisKernel = SquaredExponentialKernel

@inline squaredexponentialkernel(z::T, α::T) where {T<:AbstractFloat} = exp(-α*z)

@inline pairwisefunction(::SquaredExponentialKernel) = SquaredEuclidean()
@inline function kappa(κ::SquaredExponentialKernel{T}, z::T) where {T}
    squaredexponentialkernel(z, getvalue(κ.alpha))
end



"GammaExponentialKernel(α,γ) = exp(-α⋅‖x-y‖ᵞ)   α ∈ (0,∞), γ ∈ (0,1]"
struct GammaExponentialKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaExponentialKernel{T}(α::Real, γ::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
function GammaExponentialKernel(α::T1=1.0, γ::T2=one(T1)) where {T1<:Real,T2<:Real}
    GammaExponentialKernel{floattype(T1,T2)}(α,γ)
end

@inline gammaexponentialkernel(z::T, α::T, γ::T) where {T<:AbstractFloat} = exp(-α*z^γ)

@inline pairwisefunction(::GammaExponentialKernel) = SquaredEuclidean()
@inline function kappa(κ::GammaExponentialKernel{T}, z::T) where {T}
    gammaexponentialkernel(z, getvalue(κ.alpha), getvalue(κ.gamma))
end



"RationalQuadraticKernel(α,β) = (1 + α⋅‖x-y‖²)⁻ᵝ   α ∈ (0,∞), β ∈ (0,∞)"
struct RationalQuadraticKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    RationalQuadraticKernel{T}(α::Real, β::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,β), interval(OpenBound(zero(T)), nothing))
    )
end
function RationalQuadraticKernel(α::T1 = 1.0, β::T2 = one(T1)) where {T1<:Real,T2<:Real}
    RationalQuadraticKernel{floattype(T1,T2)}(α, β)
end

@inline rationalquadratickernel(z::T, α::T, β::T) where {T<:AbstractFloat} = (1 + α*z)^(-β)

@inline pairwisefunction(::RationalQuadraticKernel) = SquaredEuclidean()
@inline function kappa(κ::RationalQuadraticKernel{T}, z::T) where {T}
    rationalquadratickernel(z, getvalue(κ.alpha), getvalue(κ.beta))
end



"GammaRationalKernel(α,β) = (1 + α⋅‖x-y‖²ᵞ)⁻ᵝ   α ∈ (0,∞), β ∈ (0,∞), γ ∈ (0,1]"
struct GammaRationalKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaRationalKernel{T}(α::Real, β::Real, γ::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,β), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
function GammaRationalKernel(
        α::T1 = 1.0,
        β::T2 = one(T1),
        γ::T3 = one(floattype(T1,T2))
    ) where {T1<:Real,T2<:Real,T3<:Real}
    GammaRationalKernel{floattype(T1,T2,T3)}(α,β,γ)
end

@inline gammarationalkernel(z::T, α::T, β::T, γ::T) where {T<:AbstractFloat} = (1 + α*(z^γ))^(-β)

@inline pairwisefunction(::GammaRationalKernel) = SquaredEuclidean()
@inline function kappa(κ::GammaRationalKernel{T}, z::T) where {T}
    gammarationalkernel(z, getvalue(κ.alpha), getvalue(κ.beta), getvalue(κ.gamma))
end



"MaternKernel(ν,ρ) = 2ᵛ⁻¹(√(2ν)‖x-y‖²/θ)ᵛKᵥ(√(2ν)‖x-y‖²/θ)/Γ(ν)   ν ∈ (0,∞), ρ ∈ (0,∞)"
struct MaternKernel{T<:AbstractFloat} <: MercerKernel{T}
    nu::HyperParameter{T}
    rho::HyperParameter{T}
    MaternKernel{T}(ν::Real, ρ::Real) where {T<:AbstractFloat}  = new{T}(
        HyperParameter(convert(T,ν), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,ρ), interval(OpenBound(zero(T)), nothing))
    )
end
function MaternKernel(ν::T1=1.0, ρ::T2=one(T1)) where {T1<:Real,T2<:Real}
    MaternKernel{floattype(T1,T2)}(ν,ρ)
end

@inline function maternkernel(z::T, ν::T, ρ::T) where {T}
    v1 = sqrt(2ν) * z / ρ
    v1 = v1 < eps(T) ? eps(T) : v1  # Overflow risk as z -> Inf
    2 * (v1/2)^(ν) * besselk(ν, v1) / gamma(ν)
end

@inline pairwisefunction(::MaternKernel) = SquaredEuclidean()
@inline function kappa(κ::MaternKernel{T}, z::T) where {T}
    maternkernel(z, getvalue(κ.nu), getvalue(κ.rho))
end



"LinearKernel(a,c) = a⋅xᵀy + c   a ∈ (0,∞), c ∈ [0,∞)"
struct LinearKernel{T<:AbstractFloat} <: MercerKernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    LinearKernel{T}(a::Real, c::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,a), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,c), interval(ClosedBound(zero(T)), nothing))
    )
end
LinearKernel(a::T1=1.0, c::T2=one(T1)) where {T1<:Real,T2<:Real} = LinearKernel{floattype(T1,T2)}(a,c)

@inline linearkernel(z::T, a::T, c::T) where {T<:AbstractFloat} = a*z + c

@inline pairwisefunction(::LinearKernel) = ScalarProduct()
@inline kappa(κ::LinearKernel{T}, z::T) where {T} = linearkernel(z, getvalue(κ.a), getvalue(κ.c))



"PolynomialKernel(a,c,d) = (a⋅xᵀy + c)ᵈ   a ∈ (0,∞), c ∈ [0,∞), d ∈ ℤ+"
struct PolynomialKernel{T<:AbstractFloat,U<:Integer} <: MercerKernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    d::HyperParameter{U}
    function PolynomialKernel{T}(a::Real, c::Real, d::U) where {T<:AbstractFloat,U<:Integer}
        new{T,U}(HyperParameter(convert(T,a), interval(OpenBound(zero(T)), nothing)),
                 HyperParameter(convert(T,c), interval(ClosedBound(zero(T)), nothing)),
                 HyperParameter(d, interval(ClosedBound(one(U)), nothing)))
    end
end
function PolynomialKernel(a::T1=1.0, c::T2=one(T1), d::Integer=3) where {T1<:Real,T2<:Real}
    PolynomialKernel{floattype(T1,T2)}(a, c, d)
end

@inline eltypes(::Type{<:PolynomialKernel{T,U}}) where {T,U} = (T,U)
@inline thetafieldnames(κ::PolynomialKernel) = Symbol[:a, :c]

@inline polynomialkernel(z::T, a::T, c::T, d::U) where {T<:AbstractFloat,U<:Integer} = (a*z + c)^d

@inline pairwisefunction(::PolynomialKernel) = ScalarProduct()
@inline function kappa(κ::PolynomialKernel{T}, z::T) where {T}
    polynomialkernel(z, getvalue(κ.a), getvalue(κ.c), getvalue(κ.d))
end



"ExponentiatedKernel(α) = exp(α⋅xᵀy)   α ∈ (0,∞)"
struct ExponentiatedKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    ExponentiatedKernel{T}(α::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
ExponentiatedKernel(α::T1 = 1.0) where {T1<:Real} = ExponentiatedKernel{floattype(T1)}(α)

@inline exponentiatedkernel(z::T, α::T) where {T<:AbstractFloat} = exp(α*z)

@inline pairwisefunction(::ExponentiatedKernel) = ScalarProduct()
@inline kappa(κ::ExponentiatedKernel{T}, z::T) where {T} = exponentiatedkernel(z, getvalue(κ.alpha))



"PeriodicKernel(α,p) = exp(-α⋅Σⱼsin²(xⱼ-yⱼ))"
struct PeriodicKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    PeriodicKernel{T}(α::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
PeriodicKernel(α::T1 = 1.0) where {T1<:Real} = PeriodicKernel{floattype(T1)}(α)

@inline pairwisefunction(::PeriodicKernel) = SineSquared()
@inline kappa(κ::PeriodicKernel{T}, z::T) where {T} = squaredexponentialkernel(z, getvalue(κ.alpha))



#================================================
  Negative Definite Kernels
================================================#

abstract type NegativeDefiniteKernel{T<:AbstractFloat} <: Kernel{T} end
@inline isnegdef(::NegativeDefiniteKernel) = true

"PowerKernel(a,c,γ) = ‖x-y‖²ᵞ   γ ∈ (0,1]"
struct PowerKernel{T<:AbstractFloat} <: NegativeDefiniteKernel{T}
    gamma::HyperParameter{T}
    PowerKernel{T}(γ::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
PowerKernel(γ::T1 = 1.0) where {T1<:Real} = PowerKernel{floattype(T1)}(γ)

@inline powerkernel(z::T, γ::T) where {T<:AbstractFloat} = z^γ

@inline pairwisefunction(::PowerKernel) = SquaredEuclidean()
@inline kappa(κ::PowerKernel{T}, z::T) where {T} = powerkernel(z, getvalue(κ.gamma))



"LogKernel(α,γ) = log(1 + α⋅‖x-y‖²ᵞ)   α ∈ (0,∞), γ ∈ (0,1]"
struct LogKernel{T<:AbstractFloat} <: NegativeDefiniteKernel{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    LogKernel{T}(α::Real, γ::Real) where {T<:AbstractFloat} = new{T}(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
function LogKernel(α::T1 = 1.0, γ::T2 = one(T1)) where {T1<:Real,T2<:Real}
    LogKernel{floattype(T1,T2)}(α, γ)
end

@inline logkernel(z::T, α::T, γ::T) where {T<:AbstractFloat} = log(α*z^γ+1)

@inline pairwisefunction(::LogKernel) = SquaredEuclidean()
@inline function kappa(κ::LogKernel{T}, z::T) where {T}
    logkernel(z, getvalue(κ.alpha), getvalue(κ.gamma))
end


for κ in (
        ExponentialKernel,
        SquaredExponentialKernel,
        GammaExponentialKernel,
        RationalQuadraticKernel,
        GammaRationalKernel,
        MaternKernel,
        LinearKernel,
        PolynomialKernel,
        ExponentiatedKernel,
        PeriodicKernel,
        PowerKernel,
        LogKernel,
        SigmoidKernel
    )
    κ_sym = nameof(κ)
    κ_args = [:(getvalue(κ.$(θ))) for θ in fieldnames(κ)]

    @eval begin
        function ==(κ1::$(κ_sym), κ2::$(κ_sym))
            mapreduce(θ -> getfield(κ1,θ) == getfield(κ2,θ), &, fieldnames(typeof(κ1)), init = true)
        end
    end

    @eval begin
        function convert(::Type{$(κ_sym){T}}, κ::$(κ_sym)) where {T}
            $(Expr(:call, :($(κ_sym){T}), κ_args...))
        end
    end

    κs = supertype(κ)
    while κs != Any
        @eval begin
            function convert(::Type{$(nameof(κs)){T}}, κ::$(κ_sym)) where {T}
                convert($(κ_sym){T}, κ)
            end
        end
        κs = supertype(κs)
    end
end
