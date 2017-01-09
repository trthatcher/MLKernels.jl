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




function hyperparameters{T}(κ::Kernel{T})
    fields = fieldnames(κ)
    θ = Array(HyperParameter{T}, length(fields))
    for i in eachindex(fields)
        θ[i] = getfield(κ, fields[i])
    end
    return θ
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
    exponentialkernel(z, g.alpha.value)
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

@inline pairwisefunction(::RationalQuadraticKernel) = SquaredEuclidean()
@inline kappa{T}(κ::RationalQuadraticKernel{T}, z::T) = (1 + κ.alpha*z)^(-κ.beta)



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

@inline pairwisefunction(::GammaRationalKernel) = SquaredEuclidean()
@inline kappa{T}(κ::GammaRationalKernel{T}, z::T) = (1 + κ.alpha*(z^(κ.gamma)))^(-κ.beta)



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

@inline pairwisefunction(::MaternKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::MaternKernel{T}, z::T)
    v1 = sqrt(2κ.nu) * z / κ.rho
    v1 = v1 < eps(T) ? eps(T) : v1  # Overflow risk, z -> Inf
    2 * (v1/2)^(κ.nu) * besselk(κ.nu, v1) / gamma(κ.nu)
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

@inline pairwisefunction(::LinearKernel) = ScalarProduct()
@inline kappa{T}(κ::LinearKernel{T}, z::T) = κ.a*z + κ.c



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

@inline pairwisefunction(::PolynomialKernel) = ScalarProduct()
@inline kappa{T}(κ::PolynomialKernel{T}, z::T) = (κ.a*z + κ.c)^κ.d



doc"ExponentiatedKernel(α) = exp(α⋅xᵀy)   α ∈ (0,∞)"
immutable ExponentiatedKernel{T} <: MercerKernel{T}
    alpha::HyperParameter{T}
    ExponentiatedKernel(α::T) = new(HyperParameter(α, leftbounded(zero(T), :open)))
end
function ExponentiatedKernel{T1<:Real}(α::T1 = 1.0)
    T = promote_type_float(T1)
    ExponentiatedKernel{T}(convert(T,α))
end

@inline pairwisefunction(::ExponentiatedKernel) = ScalarProduct()
@inline kappa{T}(κ::ExponentiatedKernel{T}, z::T) = exp(κ.alpha*z)



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
@inline kappa{T}(κ::PeriodicKernel{T}, z::T) = exp(-κ.alpha*z)



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

@inline pairwisefunction(::PowerKernel) = SquaredEuclidean()
@inline kappa{T}(κ::PowerKernel{T}, z::T) = z^(κ.gamma)



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

@inline pairwisefunction(::LogKernel) = SquaredEuclidean()
@inline kappa{T}(κ::LogKernel{T}, z::T) = z^(κ.gamma)
