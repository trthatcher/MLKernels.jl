#===================================================================================================
  Separable Kernels
===================================================================================================#

abstract SeparableKernel{T<:FloatingPoint} <: StandardKernel{T}

# k(x,y) = ϕ(x)ᵀϕ(y)
function kernel_function{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T}, y::Array{T})
    v = kernelize_array!(κ, copy(x))
    z = kernelize_array!(κ, copy(y))
    BLAS.dot(length(v), v, 1, z, 1)
end


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

function kernelize_array!{T<:FloatingPoint}(κ::MercerSigmoidKernel{T}, z::Array{T})
    @inbounds for i = 1:length(z)
        z[i] = tanh((z[i] - κ.d)/κ.b)
    end
    z
end

isposdef_kernel(::MercerSigmoidKernel) = true

function description_string{T<:FloatingPoint}(κ::MercerSigmoidKernel{T}, eltype::Bool = true)
    "MercerSigmoidKernel" * (eltype ? "{$(T)}" : "") * "(d=$(κ.d),b=$(κ.b))"
end

function description(κ::MercerSigmoidKernel)
    print(
        """ 
         Mercer Sigmoid Kernel:

         Description to be added later.
         
        """
    )
end


#==========================================================================
  Conversions
==========================================================================#

for kernel in (:MercerSigmoidKernel,)
    @eval begin
        function convert{T<:FloatingPoint}(::Type{SeparableKernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
        function convert{T<:FloatingPoint}(::Type{StandardKernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
        function convert{T<:FloatingPoint}(::Type{SimpleKernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
        function convert{T<:FloatingPoint}(::Type{Kernel{T}}, κ::$kernel)
            convert($kernel{T}, κ)
        end
    end
end
