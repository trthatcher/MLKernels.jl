#===================================================================================================
  Kernels
===================================================================================================#

abstract Kernel{T<:AbstractFloat}

function string(κ::Kernel)
    args = [string(getvalue(getfield(κ,θ))) for θ in fieldnames(κ)]
    kernelname = typeof(κ).name.name
    string(kernelname, "(", join(args, ","), ")")
end

function show(io::IO, κ::Kernel)
    print(io, string(κ))
end

function pairwisefunction(::Kernel)
    error("No pairwise function specified for kernel")
end

@inline eltype{T}(::Kernel{T}) = T

ismercer(::Kernel) = false
isnegdef(::Kernel) = false

thetafieldnames(κ::Kernel) = fieldnames(κ)

gettheta{T}(κ::Kernel{T}) = T[gettheta(getfield(κ,θ)) for θ in thetafieldnames(κ)]

function settheta!{T}(κ::Kernel{T},v::Vector{T})
    fields = thetafieldnames(κ)
    if length(fields) != length(vector)
        throw(DimensionMismatch, "Update vector has invalid length")
    end
    for i in eachindex(fields)
        settheta!(getfield(κ, fields[i]), v[i])
    end
    return κ
end

function checktheta{T}(κ::Kernel{T},v::Vector{T})
    fields = thetafieldnames(κ)
    if length(fields) != length(vector)
        throw(DimensionMismatch, "Update vector has invalid length")
    end
    for i in eachindex(fields)
        if !checktheta!(getfield(κ, fields[i]), v[i])
            return false
        end
    end
    return true
end



#================================================
  Not True Kernels
================================================#

doc"SigmoidKernel(a,c) = tanh(a⋅xᵀy + c)   a ∈ (0,∞), c ∈ (0,∞)"
immutable SigmoidKernel{T<:AbstractFloat} <: Kernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    SigmoidKernel(a::Real, c::Real) = new(
        HyperParameter(convert(T,a), interval(OpenBound(zero(T)),   nothing)),
        HyperParameter(convert(T,c), interval(ClosedBound(zero(T)), nothing))   
    )
end
function SigmoidKernel{T1<:Real,T2<:Real}(a::T1 = 1.0, c::T2 = one(T1))
    SigmoidKernel{floattype(T1,T2)}(a,c)
end

@inline sigmoidkernel{T<:AbstractFloat}(z::T, a::T, c::T) = tanh(a*z + c)

@inline pairwisefunction(::SigmoidKernel) = ScalarProduct()
@inline kappa{T}(κ::SigmoidKernel{T}, z::T) = sigmoidkernel(z, getvalue(κ.a), getvalue(κ.c))



#================================================
  Mercer Kernels
================================================#

abstract MercerKernel{T} <: Kernel{T}
ismercer(κ::MercerKernel) = true



doc"ExponentialKernel(α) = exp(-α⋅‖x-y‖)   α ∈ (0,∞)"
immutable ExponentialKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    function ExponentialKernel(α::Real)
        new(HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)))
    end
end
ExponentialKernel{T<:Real}(α::T=1.0) = ExponentialKernel{floattype(T)}(α)
LaplacianKernel = ExponentialKernel

@inline exponentialkernel{T<:AbstractFloat}(z::T, α::T) = exp(-α*sqrt(z))

@inline pairwisefunction(::ExponentialKernel) = SquaredEuclidean()
@inline function kappa{T<:AbstractFloat}(κ::ExponentialKernel{T}, z::T)
    exponentialkernel(z, getvalue(κ.alpha))
end



doc"SquaredExponentialKernel(α) = exp(-α⋅‖x-y‖²)   α ∈ (0,∞)"
immutable SquaredExponentialKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    SquaredExponentialKernel(α::Real) = new(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
SquaredExponentialKernel{T<:Real}(α::T=1.0) = SquaredExponentialKernel{floattype(T)}(α)
GaussianKernel = SquaredExponentialKernel
RadialBasisKernel = SquaredExponentialKernel

@inline squaredexponentialkernel{T<:AbstractFloat}(z::T, α::T) = exp(-α*z)

@inline pairwisefunction(::SquaredExponentialKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::SquaredExponentialKernel{T}, z::T)
    squaredexponentialkernel(z, getvalue(κ.alpha))
end



doc"GammaExponentialKernel(α,γ) = exp(-α⋅‖x-y‖ᵞ)   α ∈ (0,∞), γ ∈ (0,1]"
immutable GammaExponentialKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaExponentialKernel(α::Real, γ::Real) = new(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
function GammaExponentialKernel{T1<:Real,T2<:Real}(α::T1=1.0, γ::T2=one(T1))
    GammaExponentialKernel{floattype(T1,T2)}(α,γ)
end

@inline gammaexponentialkernel{T<:AbstractFloat}(z::T, α::T, γ::T) = exp(-α*z^γ)

@inline pairwisefunction(::GammaExponentialKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::GammaExponentialKernel{T}, z::T)
    gammaexponentialkernel(z, getvalue(κ.alpha), getvalue(κ.gamma))
end



doc"RationalQuadraticKernel(α,β) = (1 + α⋅‖x-y‖²)⁻ᵝ   α ∈ (0,∞), β ∈ (0,∞)"
immutable RationalQuadraticKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    RationalQuadraticKernel(α::Real, β::Real) = new(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,β), interval(OpenBound(zero(T)), nothing))
    )
end
function RationalQuadraticKernel{T1<:Real,T2<:Real}(α::T1 = 1.0, β::T2 = one(T1))
    RationalQuadraticKernel{floattype(T1,T2)}(α, β)
end

@inline rationalquadratickernel{T<:AbstractFloat}(z::T, α::T, β::T) = (1 + α*z)^(-β)

@inline pairwisefunction(::RationalQuadraticKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::RationalQuadraticKernel{T}, z::T)
    rationalquadratickernel(z, getvalue(κ.alpha), getvalue(κ.beta))
end



doc"GammaRationalKernel(α,β) = (1 + α⋅‖x-y‖²ᵞ)⁻ᵝ   α ∈ (0,∞), β ∈ (0,∞), γ ∈ (0,1]"
immutable GammaRationalKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    beta::HyperParameter{T}
    gamma::HyperParameter{T}
    GammaRationalKernel(α::Real, β::Real, γ::Real) = new(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,β), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
function GammaRationalKernel{T1<:Real,T2<:Real,T3<:Real}(
        α::T1 = 1.0,
        β::T2 = one(T1),
        γ::T3 = one(floattype(T1,T2))
    )
    GammaRationalKernel{floattype(T1,T2,T3)}(α,β,γ)
end

@inline gammarationalkernel{T<:AbstractFloat}(z::T, α::T, β::T, γ::T) = (1 + α*(z^γ))^(-β)

@inline pairwisefunction(::GammaRationalKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::GammaRationalKernel{T}, z::T)
    gammarationalkernel(z, getvalue(κ.alpha), getvalue(κ.beta), getvalue(κ.gamma))
end



doc"MaternKernel(ν,ρ) = 2ᵛ⁻¹(√(2ν)‖x-y‖²/θ)ᵛKᵥ(√(2ν)‖x-y‖²/θ)/Γ(ν)   ν ∈ (0,∞), ρ ∈ (0,∞)"
immutable MaternKernel{T<:AbstractFloat} <: MercerKernel{T}
    nu::HyperParameter{T}
    rho::HyperParameter{T}
    MaternKernel(ν::Real, ρ::Real) = new(
        HyperParameter(convert(T,ν), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,ρ), interval(OpenBound(zero(T)), nothing))
    )
end
MaternKernel{T1<:Real,T2<:Real}(ν::T1=1.0, ρ::T2=one(T1)) = MaternKernel{floattype(T1,T2)}(ν,ρ)

@inline function maternkernel{T}(z::T, ν::T, ρ::T)
    v1 = sqrt(2ν) * z / ρ
    v1 = v1 < eps(T) ? eps(T) : v1  # Overflow risk as z -> Inf
    2 * (v1/2)^(ν) * besselk(ν, v1) / gamma(ν)
end

@inline pairwisefunction(::MaternKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::MaternKernel{T}, z::T)
    maternkernel(z, getvalue(κ.nu), getvalue(κ.rho))
end



doc"LinearKernel(a,c) = a⋅xᵀy + c   a ∈ (0,∞), c ∈ [0,∞)"
immutable LinearKernel{T<:AbstractFloat} <: MercerKernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    LinearKernel(a::Real, c::Real) = new(
        HyperParameter(convert(T,a), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,c), interval(ClosedBound(zero(T)), nothing))
    )
end
LinearKernel{T1<:Real,T2<:Real}(a::T1=1.0, c::T2=one(T1)) = LinearKernel{floattype(T1,T2)}(a,c)

@inline linearkernel{T<:AbstractFloat}(z::T, a::T, c::T) = a*z + c

@inline pairwisefunction(::LinearKernel) = ScalarProduct()
@inline kappa{T}(κ::LinearKernel{T}, z::T) = linearkernel(z, getvalue(κ.a), getvalue(κ.c))



doc"PolynomialKernel(a,c,d) = (a⋅xᵀy + c)ᵈ   a ∈ (0,∞), c ∈ [0,∞), d ∈ ℤ+"
immutable PolynomialKernel{T<:AbstractFloat,U<:Integer} <: MercerKernel{T}
    a::HyperParameter{T}
    c::HyperParameter{T}
    d::HyperParameter{U}
    PolynomialKernel(a::Real, c::Real, d::Integer) = new(
        HyperParameter(convert(T,a), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,c), interval(ClosedBound(zero(T)), nothing)),
        HyperParameter(convert(U,d), interval(ClosedBound(one(U)), nothing))
    )
end
function PolynomialKernel{T1<:Real,T2<:Real,U<:Integer}(a::T1 = 1.0, c::T2 = one(T1), d::U = 3)
    PolynomialKernel{floattype(T1,T2),U}(a, c, d)
end

thetafieldnames(κ::PolynomialKernel) = Symbol[:a, :c]

@inline polynomialkernel{T<:AbstractFloat,U<:Integer}(z::T, a::T, c::T, d::U) = (a*z + c)^d

@inline pairwisefunction(::PolynomialKernel) = ScalarProduct()
@inline function kappa{T}(κ::PolynomialKernel{T}, z::T)
    polynomialkernel(z, getvalue(κ.a), getvalue(κ.c), getvalue(κ.d))
end



doc"ExponentiatedKernel(α) = exp(α⋅xᵀy)   α ∈ (0,∞)"
immutable ExponentiatedKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    ExponentiatedKernel(α::Real) = new(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
ExponentiatedKernel{T1<:Real}(α::T1 = 1.0) = ExponentiatedKernel{floattype(T1)}(α)

@inline exponentiatedkernel{T<:AbstractFloat}(z::T, α::T) = exp(α*z)

@inline pairwisefunction(::ExponentiatedKernel) = ScalarProduct()
@inline kappa{T}(κ::ExponentiatedKernel{T}, z::T) = exponentiatedkernel(z, getvalue(κ.alpha))



doc"PeriodicKernel(α,p) = exp(-α⋅Σⱼsin²(xⱼ-yⱼ))"
immutable PeriodicKernel{T<:AbstractFloat} <: MercerKernel{T}
    alpha::HyperParameter{T}
    PeriodicKernel(α::Real) = new(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing))
    )
end
PeriodicKernel{T1<:Real}(α::T1 = 1.0) = PeriodicKernel{floattype(T1)}(α)

@inline pairwisefunction(::PeriodicKernel) = SineSquared()
@inline kappa{T}(κ::PeriodicKernel{T}, z::T) = squaredexponentialkernel(z, getvalue(κ.alpha))



#================================================
  Negative Definite Kernels
================================================#

abstract NegativeDefiniteKernel{T} <: Kernel{T}
isnegdef(::NegativeDefiniteKernel) = true



doc"PowerKernel(a,c,γ) = ‖x-y‖²ᵞ   γ ∈ (0,1]"
immutable PowerKernel{T<:AbstractFloat} <: NegativeDefiniteKernel{T}
    gamma::HyperParameter{T}
    PowerKernel(γ::Real) = new(
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
PowerKernel{T1<:Real}(γ::T1 = 1.0) = PowerKernel{floattype(T1)}(γ)

@inline powerkernel{T<:AbstractFloat}(z::T, γ::T) = z^γ

@inline pairwisefunction(::PowerKernel) = SquaredEuclidean()
@inline kappa{T}(κ::PowerKernel{T}, z::T) = powerkernel(z, getvalue(κ.gamma))



doc"LogKernel(α,γ) = log(1 + α⋅‖x-y‖²ᵞ)   α ∈ (0,∞), γ ∈ (0,1]"
immutable LogKernel{T<:AbstractFloat} <: NegativeDefiniteKernel{T}
    alpha::HyperParameter{T}
    gamma::HyperParameter{T}
    LogKernel(α::Real, γ::Real) = new(
        HyperParameter(convert(T,α), interval(OpenBound(zero(T)), nothing)),
        HyperParameter(convert(T,γ), interval(OpenBound(zero(T)), ClosedBound(one(T))))
    )
end
LogKernel{T1,T2}(α::T1 = 1.0, γ::T2 = one(T1)) = LogKernel{floattype(T1,T2)}(α, γ)

@inline powerkernel{T<:AbstractFloat}(z::T, α::T, γ::T) = log(α*z^γ+1)

@inline pairwisefunction(::LogKernel) = SquaredEuclidean()
@inline function kappa{T}(κ::LogKernel{T}, z::T)
    powerkernel(z, getvalue(κ.alpha), getvalue(κ.gamma))
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
    kernel_sym = κ.name.name

    @eval begin
        function ==(κ1::$kernel_sym, κ2::$kernel_sym)
            mapreduce(θ -> getfield(κ1,θ) == getfield(κ2,θ), &, true, fieldnames($kernel_sym))
        end
    end

    kernel_args = [:(getvalue(κ.$(θ))) for θ in fieldnames(κ)]
    if length(κ.parameters) == 2
        @eval begin
            function convert{T,U}(::Type{$(kernel_sym){T,U}}, κ::$(kernel_sym))
                $(Expr(:call, :($kernel_sym{T,U}), kernel_args...))
            end

            function convert{T,_,U}(::Type{$(kernel_sym){T}}, κ::$(kernel_sym){_,U})
                convert($(kernel_sym){T,U}, κ)
            end
        end
    elseif length(κ.parameters) == 1
        @eval begin
            function convert{T}(::Type{$(kernel_sym){T}}, κ::$(kernel_sym))
                $(Expr(:call, :($kernel_sym{T}), kernel_args...))
            end
        end
    else
        error("Incorrect number of parameters for code generation.")
    end
    for ψ in (
            Kernel,
            MercerKernel,
            NegativeDefiniteKernel
        )
        if κ <: ψ
            @eval begin
                function convert{T}(::Type{$(ψ.name.name){T}}, κ::$(kernel_sym))
                    convert($(kernel_sym){T}, κ)
                end
            end
        end
    end

end
