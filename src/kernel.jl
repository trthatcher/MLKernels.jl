#===================================================================================================
  Kernels
===================================================================================================#

abstract Kernel{T<:AbstractFloat}

eltype{T}(::Kernel{T}) = T 

doc"`ismercer(κ)`: returns `true` if kernel `κ` is a Mercer kernel."
ismercer(::Kernel) = false

doc"`isnegdef(κ)`: returns `true` if kernel `κ` is a continuous symmetric negative-definite kernel."
isnegdef(::Kernel) = false

doc"`attainszero(κ)`: returns `true` if ∃x,y such that κ(x,y) = 0"
attainszero(::Kernel) = true

doc"`attainspositive(κ)`: returns `true` if ∃x,y such that κ(x,y) > 0"
attainspositive(::Kernel) = true

doc"`attainsnegative(κ)`: returns `true` if ∃x,y such that κ(x,y) < 0"
attainsnegative(::Kernel) = true

doc"`isnonnegative(κ)`: returns `true` if κ(x,y) > 0 ∀x,y"
isnonnegative(κ::Kernel) = !attainsnegative(κ)

doc"`ispositive(κ)`: returns `true` if κ(x,y) ≧ 0 ∀x,y"
ispositive(κ::Kernel) = !attainsnegative(κ) && !attainszero(κ) &&  attainspositive(κ)

doc"`isnegative(κ)`: returns `true` if κ(x,y) ≦ 0 ∀x,y"
isnegative(k::Kernel) =  attainsnegative(κ) && !attainszero(κ) && !attainspositive(κ)

function show(io::IO, κ::Kernel)
    print(io, description_string(κ))
end

function convert{T<:AbstractFloat,K<:Kernel}(::Type{Kernel{T}}, ϕ::K)
    convert(K.name.primary{T}, ϕ)
end


#== Composition Class ==#

include("compositionclass.jl")


#== Standard Kernels ==#

abstract StandardKernel{T<:AbstractFloat}  <: Kernel{T}  # Either a kernel is atomic or it is a

include("pairwisekernel.jl")
include("kernelcomposition.jl")


#== Kernel Operations ==#

abstract KernelOperation{T<:AbstractFloat} <: Kernel{T}  # function of multiple kernels

include("kerneloperation.jl")
