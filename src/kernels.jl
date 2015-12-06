abstract Kernel{T}

function show(io::IO, κ::Kernel)
    print(io, description_string(κ))
end

eltype{T}(κ::Kernel{T}) = T

ismercer(::Kernel) = false
isnegdef(::Kernel) = false

kernelrange(::Kernel) = :R
# R  -> Real Number line
# Rp -> Real Non-Negative Numbers

attainszero(::Kernel) = true  # Does it attain zero?

ispositive(κ::Kernel)    = kernelrange(κ) == :Rp && !attainszero(κ)
isnonnegative(κ::Kernel) = kernelrange(κ) == :Rp


#==========================================================================
  Base Kernels
==========================================================================#

abstract BaseKernel{T<:AbstractFloat} <: Kernel{T}

include("kernels/additivekernels.jl")

immutable ARD{T<:AbstractFloat} <: BaseKernel{T}
    k::AdditiveKernel{T}
    w::Vector{T}
    function ARD(κ::AdditiveKernel{T}, w::Vector{T})
        length(w) > 0 || error("Weight vector w must be at least of length 1.")
        all(w .> 0) || error("Weight vector w must consist of positive values.")
        new(κ, w)
    end
end
ARD{T<:AbstractFloat}(κ::AdditiveKernel{T}, w::Vector{T}) = ARD{T}(κ, w)

ismercer(κ::ARD) = ismercer(κ.k)
isnegdef(κ::ARD) = isnegdef(κ.k)

kernelrange(κ::ARD) = kernelrange(κ.k)
attainszero(κ::ARD) = attainszero(κ.k)

function description_string{T<:AbstractFloat,}(κ::ARD{T}, eltype::Bool = true)
    "ARD" * (eltype ? "{$(T)}" : "") * "(κ=$(description_string(κ.k, false)),w=$(κ.w))"
end

function convert{T<:AbstractFloat}(::Type{ARD{T}}, κ::ARD)
    ARD(convert(Kernel{T},κ.k), convert(Vector{T}, κ.w))
end


#==========================================================================
  Kernel Composition
==========================================================================#

include("kernels/compositionclasses.jl")

# ψ = ϕ(κ(x,y))
immutable KernelComposition{T<:AbstractFloat,CASE} <: Kernel{T}
    phi::CompositionClass{T}
    k::Kernel{T}
    function KernelComposition(ϕ::CompositionClass{T}, κ::Kernel{T})
        iscomposable(ϕ, κ) || error("Kernel is not composable.")
        if CASE == :affine
            isa(ϕ, AffineClass) || error("Affine case flagged but composition is not affine.")
        end
        new(ϕ, κ)
    end
end
function KernelComposition{T<:AbstractFloat}(ϕ::CompositionClass{T}, κ::Kernel{T})
    KernelComposition{T, isa(ϕ, AffineClass) ? :affine : :Ø}(ϕ, κ)
end

function description_string{T<:AbstractFloat,}(κ::KernelComposition{T}, eltype::Bool = true)
    "KernelComposition" * (eltype ? "{$(T)}" : "") * "(ϕ=$(description_string(κ.phi,false))," *
    "κ=$(description_string(κ.k, false)))"
end

function convert{T<:AbstractFloat}(::Type{KernelComposition{T}}, ψ::KernelComposition)
    KernelComposition(convert(CompositionClass{T}, ψ.phi), convert(Kernel{T}, ψ.κ))
end

# Special Compositions

|>(κ::Kernel, ϕ::CompositionClass) = KernelComposition(ϕ, κ)
<|(ϕ::CompositionClass, κ::Kernel) = KernelComposition(ϕ, κ)
∘ = <|

function ^{T<:AbstractFloat}(κ::Kernel{T}, d::Integer)
    ismercer(κ) || error("Kernel must be Mercer to raise to an integer.")
    KernelComposition(PolynomialClass(one(T), zero(T), convert(T,d)), κ)
end

function ^{T<:AbstractFloat}(κ::Kernel{T}, γ::T)
    isnegdef(κ) || error("Kernel must be negative definite to raise to γ=$(γ)")
    KernelComposition(PowerClass(one(T), zero(T), γ), κ)
end

#function exp{T<:AbstractFloat}(


# Special Kernel Constructors

doc"`GaussianKernel(α)` = exp(-α⋅‖x-y‖²)"
function GaussianKernel{T<:AbstractFloat}(α::T = 1.0)
    KernelComposition(ExponentialClass(α, one(T)), SquaredDistanceKernel(one(T)))
end
SquaredExponentialKernel = GaussianKernel
RadialBasisKernel = GaussianKernel

doc"`LaplacianKernel(α)` = exp(α⋅‖x-y‖)"
function LaplacianKernel{T<:AbstractFloat}(α::T = 1.0)
    KernelComposition(ExponentialClass(α, one(T)/2), SquaredDistanceKernel(one(T)))
end

doc"`PeriodicKernel(α,p)` = exp(-α⋅Σᵢsin²(p(xᵢ-yᵢ)))"
function PeriodicKernel{T<:AbstractFloat}(α::T = 1.0, p::T = convert(T, π))
    KernelComposition(ExponentialClass(α, one(T)), SineSquaredKernel(p, one(T)))
end

doc"'RationalQuadraticKernel(α,β)` = (1 + α⋅‖x-y‖²)⁻ᵝ"
function RationalQuadraticKernel{T<:AbstractFloat}(α::T = 1.0, β::T = one(T), γ::T = one(T))
    KernelComposition(RationalQuadraticClass(α, β), SquaredDistanceKernel(one(T)))
end

doc"`MatérnKernel(ν,θ)` = 2ᵛ⁻¹(√(2ν)‖x-y‖²/θ)ᵛKᵥ(√(2ν)‖x-y‖²/θ)/Γ(ν)"
function MaternKernel{T<:AbstractFloat}(ν::T = 1.0, θ::T = one(T))
    KernelComposition(RationalQuadraticClass(α, β), SquaredDistanceKernel(one(T)))
end
MatérnKernel = MaternKernel

doc"`PolynomialKernel(α,c,d)` = (α⋅xᵀy + c)ᵈ"
function PolynomialKernel{T<:AbstractFloat}(α::T = 1.0, c = one(T), d = 3one(T))
    KernelComposition(PolynomialClass(α, c, d), ScalarProductKernel(one(T)))
end

doc"`LinearKernel(α,c,d)` = α⋅xᵀy + c"
function LinearKernel{T<:AbstractFloat}(α::T = 1.0, c = one(T))
    KernelComposition(TranslationScaleClass(α, c), ScalarProductKernel(one(T)))
end

doc"`SigmoidKernel(α,c)` = tanh(α⋅xᵀy + c)"
function SigmoidKernel{T<:AbstractFloat}(α::T = 1.0, c::T = one(T))
    KernelComposition(SigmoidClass(α, c), ScalarProductKernel(one(T)))
end


#===================================================================================================
  Composite Kernels
===================================================================================================#

abstract KernelOperator{T<:AbstractFloat} <: Kernel{T}

immutable KernelProduct{T<:AbstractFloat} <: KernelOperator{T}
    a::T
    k::Vector{Kernel{T}}
    function KernelProduct(a::T, κ::Vector{Kernel{T}})
        a > 0 || error("a = $(a) must be greater than zero.")
        if length(κ) > 1
            all(ismercer, κ) || error("All kernels must be Mercer for closure under multiplication.")
        end
        new(a, κ)
    end
end
KernelProduct{T<:AbstractFloat}(a::T, κ::Vector{Kernel{T}}) = KernelProduct{T}(a, κ)

attainszero(κ::KernelProduct) = any(attainszero, κ.k)  # Does it attain zero?
ispositive(ψ::KernelProduct)  = all(ispositive, ψ.k)


immutable KernelSum{T<:AbstractFloat} <: KernelOperator{T}
    a::T
    k::Vector{Kernel{T}}
    function KernelSum(a::T, κ::Vector{Kernel{T}})
        a >= 0 || error("a = $(a) must be greater than or equal to zero.")
        all(ismercer, κ) || all(isnegdef, κ) || error("All kernels must be Mercer or negative definite for closure under addition")
        new(a, κ)
    end
end
KernelSum{T<:AbstractFloat}(a::T, κ::Vector{Kernel{T}}) = KernelSum{T}(a, κ)

attainszero(ψ::KernelSum) = all(attainszero, ψ.k) && ψ.a == 0  # Does it attain zero?
ispositive(ψ::KernelSum)  = all(isnonnegative, ψ.k) && (ψ.a > 0 || any(ispositive, ψ.k))


for (kernel_object, kernel_op, kernel_array_op, identity) in (
        (:KernelProduct, :*, :prod, :1),
        (:KernelSum,     :+, :sum,  :0)
    )
    @eval begin

        function $kernel_object(a::Real, κ::Kernel...)
            U = promote_type(typeof(a), map(eltype, κ)...)
            $kernel_object{U}(convert(U, a), Kernel{U}[κ...])
        end

        function convert{T<:AbstractFloat}(::Type{$kernel_object{T}}, ψ::$kernel_object)
            $kernel_object(convert(T, ψ.a), Kernel{T}[ψ.k...])
        end

        isnonnegative(ψ::$kernel_object) = all(isnonnegative, ψ.k)

        ismercer(ψ::$kernel_object) = all(ismercer, ψ.k)
        isnegdef(ψ::$kernel_object) = all(isnegdef, ψ.k)

        function description_string{T<:AbstractFloat}(ψ::$kernel_object{T}, eltype::Bool = true)
            descs = map(κ -> description_string(κ, false), ψ.k)
            if eltype
                $(string(kernel_object)) * (eltype ? "{$(T)}" : "") * "($(ψ.a), $(join(descs, ", ")))"
            else
                if $kernel_op !== (*) && ψ.a != $identity
                    insert!(descs, 1, string(ψ.a))
                end
                desc_str = string("(", join(descs, $(string(" ", kernel_op, " "))), ")")
                if $kernel_op === (*) && ψ.a != $identity
                    string(ψ.a, desc_str)
                else
                    desc_str
                end
            end
        end

        $kernel_op(a::Real, κ::Kernel) = $kernel_object(a, κ)
        $kernel_op(κ::Kernel, a::Real) = $kernel_op(a, κ)

        $kernel_op(a::Real, ψ::$kernel_object) = $kernel_object($kernel_op(a, ψ.a), ψ.k...)
        $kernel_op(ψ::$kernel_object, a::Real) = $kernel_op(a, ψ)

        $kernel_op(κ1::$kernel_object, κ2::$kernel_object) = $kernel_object($kernel_op(κ1.a, κ2.a), κ1.k..., κ2.k...)

        $kernel_op(κ::Kernel, ψ::$kernel_object) = $kernel_object(ψ.a, κ, ψ.k...)
        $kernel_op(ψ::$kernel_object, κ::Kernel) = $kernel_object(ψ.a, ψ.k..., κ)

        $kernel_op(κ1::Kernel, κ2::Kernel) = $kernel_object($identity, κ1, κ2)

    end
end


#===================================================================================================
  Conversions
===================================================================================================#

for kernel in (
        concrete_subtypes(AdditiveKernel)..., 
        ARD, 
        concrete_subtypes(CompositionClass)..., 
        KernelSum,
        KernelProduct
    )
    kernel_sym = kernel.name.name  # symbol for concrete kernel type
    for parent in supertypes(kernel)
        parent_sym = parent.name.name  # symbol for abstract supertype
        @eval begin
            function convert{T<:AbstractFloat}(::Type{$parent_sym{T}}, κ::$kernel_sym)
                convert($kernel_sym{T}, κ)
            end
        end
    end
end
