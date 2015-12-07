#===================================================================================================
  Kernels
===================================================================================================#

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


#===================================================================================================
  Base Kernels
===================================================================================================#

abstract BaseKernel{T<:AbstractFloat} <: Kernel{T}


#==========================================================================
  Additive Kernel: k(x,y) = sum(k(x_i,y_i))    x ∈ ℝⁿ, y ∈ ℝⁿ
==========================================================================#

abstract AdditiveKernel{T<:AbstractFloat} <: BaseKernel{T}

include("kernels/additivekernels.jl")

for kernel_obj in concrete_subtypes(AdditiveKernel)
    kernel_name = kernel_obj.name.name  # symbol for concrete kernel type
    field_conversions = [:(convert(T, κ.$field)) for field in fieldnames(kernel_obj)]
    constructorcall = Expr(:call, kernel_name, field_conversions...)
    @eval begin
        convert{T<:AbstractFloat}(::Type{$kernel_name{T}}, κ::$kernel_name) = $constructorcall
    end
end


#==========================================================================
  ARD Kernel
==========================================================================#

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
  Composition Classes
==========================================================================#

abstract CompositionClass{T<:AbstractFloat}

iscomposable(::CompositionClass, ::Kernel) = true
ismercer(ϕ::CompositionClass, ::Kernel) = ismercer(ϕ)
kernelrange(ϕ::CompositionClass, ::Kernel) = kernelrange(ϕ)
attainszero(ϕ::CompositionClass, ::Kernel) = attainszero(ϕ)

include("kernels/compositionclasses.jl")

for class_obj in concrete_subtypes(CompositionClass)
    class_name = class_obj.name.name  # symbol for concrete kernel type
    field_conversions = [:(convert(T, κ.$field)) for field in fieldnames(class_obj)]
    constructorcall = Expr(:call, class_name, field_conversions...)
    @eval begin
        convert{T<:AbstractFloat}(::Type{$class_name{T}}, κ::$class_name) = $constructorcall
    end
end


#==========================================================================
  Kernel Composition ψ = ϕ(κ(x,y))
==========================================================================#

immutable KernelComposition{T<:AbstractFloat} <: Kernel{T}
    phi::CompositionClass{T}
    k::Kernel{T}
    function KernelComposition(ϕ::CompositionClass{T}, κ::Kernel{T})
        iscomposable(ϕ, κ) || error("Kernel is not composable.")
        new(ϕ, κ)
    end
end
function KernelComposition{T<:AbstractFloat}(ϕ::CompositionClass{T}, κ::Kernel{T})
    KernelComposition{T}(ϕ, κ)
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

function exp{T<:AbstractFloat}(κ::Kernel{T})
    ismercer(κ) || error("Kernel must be Mercer to exponentiate.")
    KernelComposition(ExponentiatedClass(one(T), zero(T)), κ)
end


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

doc"`PolynomialKernel(a,c,d)` = (a⋅xᵀy + c)ᵈ"
function PolynomialKernel{T<:AbstractFloat}(a::T = 1.0, c = one(T), d = 3one(T))
    KernelComposition(PolynomialClass(a, c, d), ScalarProductKernel{T}())
end

doc"`LinearKernel(α,c,d)` = α⋅xᵀy + c"
function LinearKernel{T<:AbstractFloat}(α::T = 1.0, c = one(T))
    KernelComposition(TranslationScaleClass(α, c), ScalarProductKernel{T}())
end

doc"`SigmoidKernel(α,c)` = tanh(α⋅xᵀy + c)"
function SigmoidKernel{T<:AbstractFloat}(α::T = 1.0, c::T = one(T))
    KernelComposition(SigmoidClass(α, c), ScalarProductKernel{T}())
end


#===================================================================================================
  Kernel Operations
===================================================================================================#

abstract KernelOperation{T<:AbstractFloat} <: Kernel{T}

#==========================================================================
  Kernel Affinity
==========================================================================#

doc"KernelAffinity(κ;a,c) = a⋅κ + c"
immutable KernelAffinity{T<:AbstractFloat} <: KernelOperation{T}
    a::T
    c::T
    k::Kernel{T}
    function KernelAffinity(a::T, c::T, κ::Kernel{T})
        a > 0 || error("a = $(a) must be greater than zero.")
        c >= 0 || error("c = $(c) must be greater than or equal to zero.")
        new(a, c, κ)
    end
end
KernelAffinity{T<:AbstractFloat}(a::T, c::T, κ::Kernel{T}) = KernelAffinity{T}(a, c, κ)

ismercer(ψ::KernelAffinity)    = ismercer(ψ.k)
kernelrange(ψ::KernelAffinity) = kernelrange(ψ.k)
attainszero(ψ::KernelAffinity) = attainszero(ψ.k)

function description_string{T<:AbstractFloat}(ψ::KernelAffinity{T}, eltype::Bool = true)
    "Affine" * (eltype ? "{$(T)}" : "") * "(a=$(ψ.a),c=$(ψ.c),κ=" * 
    description_string(ψ.k) * ")"
end

function convert{T<:AbstractFloat}(::Type{KernelAffinity{T}}, ψ::KernelAffinity)
    KernelAffinity(convert(T, ψ.a), convert(T, ψ.c), convert(Kernel{T}, ψ.k))
end

@inline phi{T<:AbstractFloat}(ψ::KernelAffinity{T}, z::T) = ψ.a*z + ψ.c

# Operations

+{T<:AbstractFloat}(κ::Kernel{T}, c::Real) = KernelAffinity(one(T), convert(T, c), κ)
+(c::Real, κ::Kernel) = +(κ, c)

*{T<:AbstractFloat}(κ::Kernel{T}, a::Real) = KernelAffinity(convert(T, a), zero(T), κ)
*(a::Real, κ::Kernel) = +(κ, a)

+{T<:AbstractFloat}(κ::KernelAffinity{T}, c::Real) = KernelAffinity(κ.a, κ.c + convert(T, c), κ.k)
+(c::Real, κ::KernelAffinity) = +(κ, c)

function *{T<:AbstractFloat}(κ::KernelAffinity{T}, a::Real)
    a = convert(T, a)
    KernelAffinity(a * κ.a, a * κ.c, κ.k)
end
*(a::Real, κ::KernelAffinity) = +(κ, a)

function ^{T<:AbstractFloat}(ψ::KernelAffinity{T}, d::Integer)
    ismercer(ψ.k) || error("Kernel must be Mercer to raise to an integer.")
    KernelComposition(PolynomialClass(ψ.a, ψ.c, convert(T,d)), ψ.k)
end

function ^{T<:AbstractFloat}(ψ::KernelAffinity{T}, γ::AbstractFloat)
    isnegdef(ψ.k) || error("Kernel must be negative definite to raise to γ=$(γ)")
    KernelComposition(PowerClass(ψ.a, ψ.c, convert(T,γ)), ψ.k)
end

function exp{T<:AbstractFloat}(ψ::KernelAffinity{T})
    ismercer(ψ.k) || error("Kernel must be Mercer to raise to exponentiate.")
    KernelComposition(ExponentiatedClass(ψ.a, ψ.c), ψ.k)
end


#==========================================================================
  Kernel Product
==========================================================================#

#=
immutable KernelProduct{T<:AbstractFloat} <: KernelOperation{T}
    a::T
    k::Vector{Kernel{T}}
    function KernelProduct(a::T, κ::Vector{Kernel{T}})
        a > 0 || error("a = $(a) must be greater than zero.")
        if length(κ) > 1
            all(ismercer, κ) || error("Kernels must be Mercer for closure under multiplication.")
        end
        new(a, κ)
    end
end
KernelProduct{T<:AbstractFloat}(a::T, κ::Vector{Kernel{T}}) = KernelProduct{T}(a, κ)

attainszero(κ::KernelProduct) = any(attainszero, κ.k)  # Does it attain zero?
ispositive(ψ::KernelProduct)  = all(ispositive, ψ.k)


immutable KernelSum{T<:AbstractFloat} <: KernelOperation{T}
    c::T
    k::Vector{Kernel{T}}
    function KernelSum(c::T, κ::Vector{Kernel{T}})
        c >= 0 || error("c = $(c) must be greater than or equal to zero.")
        if !(all(ismercer, κ) || all(isnegdef, κ))
            error("All kernels must be Mercer or negative definite for closure under addition")
        end
        new(c, κ)
    end
end
KernelSum{T<:AbstractFloat}(c::T, κ::Vector{Kernel{T}}) = KernelSum{T}(c, κ)

attainszero(ψ::KernelSum) = all(attainszero, ψ.k) && ψ.c == 0  # Does it attain zero?
ispositive(ψ::KernelSum)  = all(isnonnegative, ψ.k) && (ψ.c > 0 || any(ispositive, ψ.k))


for (kernel_object, kernel_op, kernel_array_op, identity, variable) in (
        (:KernelProduct, :*, :prod, :1, :a),
        (:KernelSum,     :+, :sum,  :0, :c)
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
                $(string(kernel_object)) * (eltype ? "{$(T)}" : "") * "($(variable)=$(ψ.a), $(join(descs, ", ")))"
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

=#
#===================================================================================================
  Conversions
===================================================================================================#

#=
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
=#
