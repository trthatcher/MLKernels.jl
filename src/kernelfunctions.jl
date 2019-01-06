# Kernel Functions =========================================================================

abstract type Kernel{T<:AbstractFloat} end

function string(κ::Kernel{T}) where {T}
    args = [string(getfield(κ,θ)) for θ in fieldnames(typeof(κ))]
    kernelname = typeof(κ).name.name
    string(kernelname, "{", string(T), "}(", join(args, ","), ")")
end

function show(io::IO, κ::Kernel)
    print(io, string(κ))
end

@inline eltype(::Type{<:Kernel{E}}) where {E} = E
@inline eltype(κ::Kernel) = eltype(typeof(κ))

"""
    ismercer(κ::Kernel)

Returns `true` if kernel `κ` is a Mercer kernel; `false` otherwise.
"""
ismercer(::Kernel) = false

"""
    isnegdef(κ::Kernel)

Returns `true` if the kernel `κ` is a negative definite kernel; `false` otherwise.
"""
isnegdef(::Kernel) = false

"""
    isstationary(κ::Kernel)

Returns `true` if the kernel `κ` is a stationary kernel; `false` otherwise.
"""
isstationary(κ::Kernel) = isstationary(basefunction(κ))

"""
    isisotropic(κ::Kernel)

Returns `true` if the kernel `κ` is an isotropic kernel; `false` otherwise.
"""
isisotropic(κ::Kernel)  = isisotropic(basefunction(κ))


# Mercer Kernels ===========================================================================

abstract type MercerKernel{T<:AbstractFloat} <: Kernel{T} end

@inline ismercer(::MercerKernel) = true

const mercer_kernels = [
    "exponential",
    "exponentiated",
    "rationalquadratic",
    "matern",
    "polynomial"
]

for kname in mercer_kernels
    include(joinpath("kernelfunctions", "mercer", "$(kname).jl"))
end


# Negative Definite Kernels ================================================================

abstract type NegativeDefiniteKernel{T<:AbstractFloat} <: Kernel{T} end

@inline isnegdef(::NegativeDefiniteKernel) = true

const negdef_kernels = [
    "log",
    "power"
]

for kname in negdef_kernels
    include(joinpath("kernelfunctions", "negativedefinite", "$(kname).jl"))
end


# Other Kernels ============================================================================

const other_kernels = [
    "sigmoid"
]

for kname in other_kernels
    include(joinpath("kernelfunctions", "$(kname).jl"))
end