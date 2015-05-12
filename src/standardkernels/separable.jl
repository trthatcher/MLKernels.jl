#===================================================================================================
  Separable Kernels
===================================================================================================#

#== Mercer Sigmoid Kernel ===============#

immutable MercerSigmoidKernel{T<:FloatingPoint} <: SeparableKernel{T}
    d::T
    b::T
    function MercerSigmoidKernel(d::T, b::T)
        b > 0 || throw(ArgumentError("b = $(b) must be a positive number."))
        new(d, b)
    end
end
MercerSigmoidKernel{T<:FloatingPoint}(d::T = 0.0, b::T = one(T)) = MercerSigmoidKernel{T}(d, b)

function convert{T<:FloatingPoint}(::Type{MercerSigmoidKernel{T}}, κ::MercerSigmoidKernel)
    MercerSigmoidKernel(convert(T, κ.d), convert(T, κ.b))
end

kappa{T<:FloatingPoint}(κ::MercerSigmoidKernel{T}, z::T) = tanh((z - κ.d)/κ.b)
kappa_dz{T<:FloatingPoint}(κ::MercerSigmoidKernel{T}, z::T) = (1 - kappa(κ,z)^2) / κ.b
kappa_db{T<:FloatingPoint}(κ::MercerSigmoidKernel{T}, z::T) = -(1 - kappa(κ,z)^2) * (z - κ.d) * κ.b^-2
kappa_dd{T<:FloatingPoint}(κ::MercerSigmoidKernel{T}, z::T) = (kappa(κ,z)^2 - 1) / κ.b

function kappa_dp{T<:FloatingPoint}(κ::MercerSigmoidKernel{T}, param::Symbol, z::T)
    param == :b ? kappa_db(κ, z) :
    param == :d ? kappa_dd(κ, z) :
                  zero(T)
end

isposdef(::MercerSigmoidKernel) = true

function description_string{T<:FloatingPoint}(κ::MercerSigmoidKernel{T}, eltype::Bool = true)
    "MercerSigmoidKernel" * (eltype ? "{$(T)}" : "") * "(d=$(κ.d),b=$(κ.b))"
end

function description_string_long(::MercerSigmoidKernel)
    """ 
    Mercer Sigmoid Kernel:

    Description to be added later.     
    """
end


#==========================================================================
  Conversions
==========================================================================#

for kernelobject in (:MercerSigmoidKernel,)
    for kerneltype in (:SeparableKernel, :StandardKernel, :SimpleKernel, :Kernel)
        @eval begin
            function convert{T<:FloatingPoint}(::Type{$kerneltype{T}}, κ::$kernelobject)
                convert($kernelobject{T}, κ)
            end
        end
    end
end
