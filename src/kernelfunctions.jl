# Kernel Functions =========================================================================

abstract type Kernel{T<:AbstractFloat} end

function string(κ::Kernel)
    args = [string(getvalue(getfield(κ,θ))) for θ in fieldnames(typeof(κ))]
    kernelname = typeof(κ).name.name
    string(kernelname, "(", join(args, ","), ")")
end

function show(io::IO, κ::Kernel)
    print(io, string(κ))
end

function basefunction(::Kernel)
    error("No base function specified for kernel")
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

thetafieldnames(κ::Kernel) = fieldnames(typeof(κ))

gettheta(κ::Kernel{T}) where {T} = T[gettheta(getfield(κ,θ)) for θ in thetafieldnames(κ)]

function settheta!(κ::Kernel{T},v::Vector{T}) where {T}
    fields = thetafieldnames(κ)
    if length(fields) != length(v)
        throw(DimensionMismatch("Update vector has invalid length"))
    end
    for i in eachindex(fields)
        settheta!(getfield(κ, fields[i]), v[i])
    end
    return κ
end

function checktheta(κ::Kernel{T},v::Vector{T}) where {T}
    fields = thetafieldnames(κ)
    if length(fields) != length(v)
        throw(DimensionMismatch("Update vector has invalid length"))
    end
    for i in eachindex(fields)
        if !checktheta(getfield(κ, fields[i]), v[i])
            return false
        end
    end
    return true
end

function floattype(T_i::DataType...)
    T_max = promote_type(T_i...)
    T_max <: AbstractFloat ? T_max : Float64
end


# Mercer Kernels ===========================================================================

abstract type MercerKernel{T<:AbstractFloat} <: Kernel{T} end
@inline ismercer(::MercerKernel) = true

const mercer_kernels = [
    "exponential",
    "squaredexponential",
    "gammaexponential",
    "rationalquadratic",
    "gammarational",
    "matern",
    "linear",
    "polynomial",
    "exponentiated",
    "periodic"
    ]

for kname in mercer_kernels
    include(joinpath("kernelfunctions", "mercer", "$(kname).jl"))
end


# Negative Definite Kernels ================================================================

abstract type NegativeDefiniteKernel{T<:AbstractFloat} <: Kernel{T} end
@inline isnegdef(::NegativeDefiniteKernel) = true

const negdef_kernels = [
    "power",
    "log"
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

for κ in [
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
    ]
    κ_sym = nameof(κ)
    κ_args = [:(getvalue(κ.$(θ))) for θ in fieldnames(κ)]

    @eval begin
        function ==(κ1::$(κ_sym), κ2::$(κ_sym))
            mapreduce(θ -> getfield(κ1,θ) == getfield(κ2,θ), &, fieldnames(typeof(κ1)), init = true)
        end
    end

    @eval begin
        function convert(::Type{$(κ_sym){T}}, κ::$(κ_sym)) where {T}
            $(Expr(:call, :($(κ_sym){T}), κ_args...))
        end
    end

    κs = supertype(κ)
    while κs != Any
        @eval begin
            function convert(::Type{$(nameof(κs)){T}}, κ::$(κ_sym)) where {T}
                convert($(κ_sym){T}, κ)
            end
        end
        κs = supertype(κs)
    end
end
