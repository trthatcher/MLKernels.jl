#===================================================================================================
  Kernels
===================================================================================================#

abstract Kernel{T<:AbstractFloat}

function show(io::IO, κ::Kernel)
    print(io, description_string(κ))
end

function description_string(κ::Kernel)
    sym_map = Dict(
        :alpha => :α,
        :beta  => :β,
        :gamma => :γ,
        :nu    => :ν,
        :rho   => :ρ
    )
    args = ["$(get(sym_map, θ, θ))=$(getfield(κ,θ).value)" for θ in fieldnames(κ)]
    kernelname = typeof(κ).name.name
    string(kernelname, "(", join(args, ","), ")")
end

function pairwisefunction(::Kernel)
    error("No pairwise function specified for kernel")
end

@inline eltype{T}(::Kernel{T}) = T


doc"SigmoidKernel(a,c) = tanh(a⋅xᵀy + c)   a ∈ (0,∞), c ∈ (0,∞)"
immutable SigmoidKernel{T} <: Kernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    SigmoidKernel(a::T, c::T) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed))   
    )
end
function SigmoidKernel{T1<:Real,T2<:Real}(
        a::T1 = 1.0,
        c::T2 = one(T1)
    )
    T = promote_type_float(T1, T2)
    SigmoidKernel{T}(convert(T,a), convert(T,c))
end


@inline sigmoidkernel{T<:AbstractFloat}(z::T, a::T, c::T) = tanh(a*z + c)

@inline pairwisefunction(::SigmoidKernel) = ScalarProduct()
@inline kappa{T}(κ::SigmoidKernel{T}, z::T) = sigmoidkernel(z, κ.a.value, κ.c.value)



#================================================
  Mercer Kernels
================================================#

abstract MercerKernel{T} <: Kernel{T}
ismercer(κ::MercerKernel) = true



doc"ExponentialKernel(α) = exp(-α⋅‖x-y‖)   α ∈ (0,∞)"
immutable ExponentialKernel{T} <: MercerKernel{T}
    alpha::HyperParameter{T}
    ExponentialKernel(α::T) = new(
        HyperParameter(α, leftbounded(zero(T), :open))
    )
end
function ExponentialKernel{T1<:Real}(α::T1 = 1.0)
    T = promote_type_float(T1)
    ExponentialKernel{T}(convert(T,α))
end
LaplacianKernel = ExponentialKernel

@inline exponentialkernel{T<:AbstractFloat}(z::T, α::T) = exp(-α*sqrt(z))

@inline pairwisefunction(::ExponentialKernel) = Euclidean()
@inline function kappa{T<:AbstractFloat}(κ::ExponentialKernel{T}, z::T)
    exponentialkernel(z, κ.alpha.value)
end



doc"SquaredExponentialKernel(α) = exp(-α⋅‖x-y‖²)   α ∈ (0,∞)"
immutable SquaredExponentialKernel{T} <: MercerKernel{T}
    alpha::HyperParameter{T}
    SquaredExponentialKernel(α::T) = new(
        HyperParameter(α, leftbounded(zero(T), :open))
    )
end
function SquaredExponentialKernel{T1<:Real}(α::T1 = 1.0)
    T = promote_type_float(T1)
    SquaredExponentialKernel{T}(convert(T,α))
end
GaussianKernel = SquaredExponentialKernel
RadialBasisKernel = SquaredExponentialKernel

@inline squaredexponentialkernel{T<:AbstractFloat}(z::T, α::T) = exp(-α*z)

@inline pairwisefunction(::SquaredExponentialKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::SquaredExponentialKernel{T}, z::T)
    squaredexponentialkernel(z, κ.alpha.value)
end



doc"GammaExponentialKernel(α,γ) = exp(-α⋅‖x-y‖ᵞ)   α ∈ (0,∞), γ ∈ (0,1]"
immutable GammaExponentialKernel{T} <: MercerKernel{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaExponentialKernel(α::T, γ::T) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(γ, Interval(Bound(zero(T), :open), Bound(one(T), :closed)))
    )
end
function GammaExponentialKernel{T1<:Real,T2<:Real}(
        α::T1 = 1.0,
        γ::T2 = one(T1)
    )
    T = promote_type_float(T1, T2)
    GammaExponentialKernel{T}(convert(T,α), convert(T,γ))
end

@inline gammaexponentialkernel{T<:AbstractFloat}(z::T, α::T, γ::T) = exp(-α*z^γ)

@inline pairwisefunction(::GammaExponentialKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::GammaExponentialKernel{T}, z::T)
    gammaexponentialkernel(z, κ.alpha.value, κ.gamma.value)
end



doc"RationalQuadraticKernel(α,β) = (1 + α⋅‖x-y‖²)⁻ᵝ   α ∈ (0,∞), β ∈ (0,∞)"
immutable RationalQuadraticKernel{T} <: MercerKernel{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    RationalQuadraticKernel(α::T, β::T) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(β, leftbounded(zero(T), :open))
    )
end
function RationalQuadraticKernel{T1<:Real,T2<:Real}(
        α::T1 = 1.0,
        β::T2 = one(T1)
    )
    T = promote_type_float(T1, T2)
    RationalQuadraticKernel{T}(convert(T,α), convert(T,β))
end

@inline rationalquadratickernel{T<:AbstractFloat}(z::T, α::T, β::T) = (1 + α*z)^(-β)

@inline pairwisefunction(::RationalQuadraticKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::RationalQuadraticKernel{T}, z::T)
    rationalquadratickernel(z, κ.alpha.value, κ.beta.value)
end



doc"GammaRationalKernel(α,β) = (1 + α⋅‖x-y‖²ᵞ)⁻ᵝ   α ∈ (0,∞), β ∈ (0,∞), γ ∈ (0,∞)"
immutable GammaRationalKernel{T} <: MercerKernel{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaRationalKernel(α::T, β::T, γ::T) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(β, leftbounded(zero(T), :open)),
        HyperParameter(γ, Interval(Bound(zero(T), :open), Bound(one(T), :closed)))
    )
end
function GammaRationalKernel{T1<:Real,T2<:Real,T3<:Real}(
        α::T1 = 1.0,
        β::T2 = one(T1),
        γ::T3 = one(promote_type_float(T1,T2))
    )
    T = promote_type_float(T1, T2, T3)
    GammaRationalKernel{T}(convert(T,α), convert(T,β), convert(T,γ))
end

@inline gammarationalkernel{T<:AbstractFloat}(z::T, α::T, β::T, γ::T) = (1 + α*(z^γ))^(-β)

@inline pairwisefunction(::GammaRationalKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::GammaRationalKernel{T}, z::T)
    gammarationalkernel(z, κ.alpha.value, κ.beta.value, κ.gamma.value)
end



doc"MaternKernel(ν,ρ) = 2ᵛ⁻¹(√(2ν)‖x-y‖²/θ)ᵛKᵥ(√(2ν)‖x-y‖²/θ)/Γ(ν)   ν ∈ (0,∞), ρ ∈ (0,∞)"
immutable MaternKernel{T} <: MercerKernel{T}
    nu::HyperParameter{T}
    rho::HyperParameter{T}
    MaternKernel(ν::T, ρ::T) = new(
        HyperParameter(ν, leftbounded(zero(T), :open)),
        HyperParameter(ρ, leftbounded(zero(T), :open))
    )
end
function MaternKernel{T1<:Real,T2<:Real}(
        ν::T1 = 1.0,
        ρ::T2 = one(T1)
    )
    T = promote_type_float(T1, T2)
    MaternKernel{T}(convert(T,ν), convert(T,ρ))
end

@inline function maternkernel{T}(z::T, ν::T, ρ::T)
    v1 = sqrt(2ν) * z / ρ
    v1 = v1 < eps(T) ? eps(T) : v1  # Overflow risk, z -> Inf
    2 * (v1/2)^(ν) * besselk(ν, v1) / gamma(ν)
end

@inline pairwisefunction(::MaternKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::MaternKernel{T}, z::T)
    maternkernel(z, κ.nu.value, κ.rho.value)
end



doc"LinearKernel(a,c) = a⋅xᵀy + c   a ∈ (0,∞), c ∈ [0,∞)"
immutable LinearKernel{T} <: MercerKernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    LinearKernel(a::T, c::T) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed))
    )
end
function LinearKernel{T1<:Real,T2<:Real}(
        a::T1 = 1.0,
        c::T2 = one(T1)
    )
    T = promote_type_float(T1, T2)
    LinearKernel{T}(convert(T,a), convert(T,c))
end

@inline linearkernel{T<:AbstractFloat}(z::T, a::T, c::T) = a*z + c

@inline pairwisefunction(::LinearKernel) = ScalarProduct()
@inline kappa{T}(κ::LinearKernel{T}, z::T) = linearkernel(z, κ.a.value, κ.c.value)



doc"PolynomialKernel(a,c,d) = (a⋅xᵀy + c)ᵈ   a ∈ (0,∞), c ∈ [0,∞), d ∈ ℤ+"
immutable PolynomialKernel{T,U<:Integer} <: MercerKernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    d::HyperParameter{U}
    PolynomialKernel(a::T, c::T, d::U) = new(
        HyperParameter(a, leftbounded(zero(T), :open)),
        HyperParameter(c, leftbounded(zero(T), :closed)),
        HyperParameter(d, leftbounded(one(U),  :closed))
    )
end
function PolynomialKernel{T1<:Real,T2<:Real,U<:Integer}(
        a::T1 = 1.0,
        c::T2 = one(T1),
        d::U = 3
    )
    T = promote_type_float(T1, T2)
    PolynomialKernel{T,U}(convert(T,a), convert(T,c), d)
end

@inline polynomialkernel{T<:AbstractFloat,U<:Integer}(z::T, a::T, c::T, d::U) = (a*z + c)^d

@inline pairwisefunction(::PolynomialKernel) = ScalarProduct()
@inline function kappa{T}(κ::PolynomialKernel{T}, z::T)
    polynomialkernel(z, κ.a.value, κ.c.value, κ.d.value)
end



doc"ExponentiatedKernel(α) = exp(α⋅xᵀy)   α ∈ (0,∞)"
immutable ExponentiatedKernel{T} <: MercerKernel{T}
    alpha::HyperParameter{T}
    ExponentiatedKernel(α::T) = new(HyperParameter(α, leftbounded(zero(T), :open)))
end
function ExponentiatedKernel{T1<:Real}(α::T1 = 1.0)
    T = promote_type_float(T1)
    ExponentiatedKernel{T}(convert(T,α))
end

@inline exponentiatedkernel{T<:AbstractFloat}(z::T, α::T) = exp(α*z)

@inline pairwisefunction(::ExponentiatedKernel) = ScalarProduct()
@inline kappa{T}(κ::ExponentiatedKernel{T}, z::T) = exponentiatedkernel(z, κ.alpha.value)



doc"PeriodicKernel(α,p) = exp(-α⋅Σⱼsin²(xⱼ-yⱼ))"
immutable PeriodicKernel{T} <: MercerKernel{T}
    alpha::HyperParameter{T}
    PeriodicKernel(α::T) = new(
        HyperParameter(α, leftbounded(zero(T), :open))
    )
end
function PeriodicKernel{T1<:Real}(α::T1 = 1.0)
    T = promote_type_float(T1)
    PeriodicKernel{T}(convert(T,α))
end

@inline pairwisefunction(::PeriodicKernel) = SquaredSine()
@inline kappa{T}(κ::PeriodicKernel{T}, z::T) = squaredexponentialkernel(z, κ.alpha.value)



#================================================
  Negative Definite Kernels
================================================#

abstract NegativeDefiniteKernel{T} <: Kernel{T}



doc"PowerKernel(a,c,γ) = ‖x-y‖²ᵞ   γ ∈ (0,1]"
immutable PowerKernel{T} <: NegativeDefiniteKernel{T}
    gamma::HyperParameter{T}
    PowerKernel(γ::T) = new(
        HyperParameter(γ, Interval(Bound(zero(T), :open), Bound(one(T), :closed)))
    )
end
function PowerKernel{T1<:Real}(γ::T1 = 1.0)
    T = promote_type_float(T1)
    PowerKernel{T}(convert(T,γ))
end

@inline powerkernel{T<:AbstractFloat}(z::T, γ::T) = z^γ

@inline pairwisefunction(::PowerKernel) = SquaredEuclidean()
@inline kappa{T}(κ::PowerKernel{T}, z::T) = powerkernel(z, κ.gamma.value)


doc"LogKernel(α,γ) = log(1 + α⋅‖x-y‖²ᵞ)   α ∈ (0,∞), γ ∈ (0,1]"
immutable LogKernel{T} <: NegativeDefiniteKernel{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    LogKernel(α::T, γ::T) = new(
        HyperParameter(α, leftbounded(zero(T), :open)),
        HyperParameter(γ, Interval(Bound(zero(T), :open), Bound(one(T), :closed)))
    )
end
function LogKernel{T1,T2}(
        α::T1 = 1.0,
        γ::T2 = one(T1)
    )
    T = promote_type_float(T1, T2)
    LogKernel{T}(convert(T,α), convert(T,γ))
end

@inline powerkernel{T<:AbstractFloat}(z::T, α::T, γ::T) = log(α*z^γ+1)

@inline pairwisefunction(::LogKernel) = SquaredEuclidean()
@inline kappa{T}(κ::LogKernel{T}, z::T) = powerkernel(z, κ.alpha.value, κ.gamma.value)
