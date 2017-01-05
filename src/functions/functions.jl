#===================================================================================================
  RealKernels
===================================================================================================#

abstract MathematicalKernel{T}

abstract RealKernel{T<:AbstractFloat} <: MathematicalKernel{T}

eltype{T}(::RealKernel{T}) = T 

doc"`ismercer(f)`: returns `true` if `f` is a valid real-valued Mercer kernel"
ismercer(::RealKernel) = false

doc"`isnegdef(f)`: returns `true` if `f` is a valid real-valued negative-definite kernel"
isnegdef(::RealKernel) = false

doc"`ismetric(f)`: returns `true` if `f` is a valid metric"
ismetric(::RealKernel) = false

doc"`isinnerprod(f)`: returns `true` if `f` is a valid inner product"
isinnerprod(::RealKernel) = false

doc"`attainszero(f)`: returns `true` if ∃x,y,θ such that f(x,y;θ) = 0"
attainszero(::RealKernel) = true

doc"`attainspositive(f)`: returns `true` if ∃x,y,θ such that f(x,y;θ) > 0"
attainspositive(::RealKernel) = true

doc"`attainsnegative(f)`: returns `true` if ∃x,y,θ such that f(x,y;θ) < 0"
attainsnegative(::RealKernel) = true

doc"`isnonnegative(f)`: returns `true` if f(x,y;θ) > 0 ∀x,y,θ"
isnonnegative(f::RealKernel) = !attainsnegative(f)

doc"`ispositive(f)`: returns `true` if f(x,y;θ) ≧ 0 ∀x,y,θ"
ispositive(f::RealKernel) = !attainsnegative(f) && !attainszero(f) &&  attainspositive(f)

doc"`isnegative(f)`: returns `true` if f(x,y;θ) ≦ 0 ∀x,y,θ"
isnegative(f::RealKernel) =  attainsnegative(f) && !attainszero(f) && !attainspositive(f)

function show(io::IO, f::RealKernel)
    print(io, description_string(f))
end

function convert{T<:AbstractFloat,F<:RealKernel}(::Type{RealKernel{T}}, f::F)
    convert(F.name.primary{T}, f)
end

# Kernels

include("functions/pairwisefunction.jl")
include("functions/compositefunction.jl")
include("functions/pointwisefunction.jl")
