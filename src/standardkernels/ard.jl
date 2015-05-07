typealias ARDKernelTypes{T<:FloatingPoint} Union(SquaredDistanceKernel{T}, ScalarProductKernel{T})

immutable ARD{T<:FloatingPoint,K<:StandardKernel{T}} <: StandardKernel{T}
    kernel::K
    weights::Vector{T}
    function ARD(k::K, weights::Vector{T})
        isa(k, ARDKernelTypes) || throw(ArgumentError("ARD only implemented for $(join(ARDKernelTypes.body.types, ", ", " and "))"))
        all(weights .>= 0) || throw(ArgumentError("weights = $(weights) must all be >= 0."))
        new(k, weights)
    end
end

ARD{T<:FloatingPoint}(kernel::ARDKernelTypes{T}, weights::Vector{T}) = ARD{T,typeof(kernel)}(kernel, weights)
ARD{T<:FloatingPoint}(kernel::ARDKernelTypes{T}, dim::Integer) = ARD{T,typeof(kernel)}(kernel, ones(T, dim))

function description_string{T<:FloatingPoint,K<:StandardKernel}(κ::ARD{T,K}, eltype::Bool = true)
    "ARD" * (eltype ? "{$(T)}" : "") * "(kernel=$(description_string(κ.kernel, false)), weights=$(κ.weights))"
end

kernel{T<:FloatingPoint,K<:SquaredDistanceKernel}(κ::ARD{T,K}, x::Array{T}, y::Array{T}) = kappa(κ.kernel, sqdist(x, y, κ.weights))
kernel{T<:FloatingPoint,K<:ScalarProductKernel}(κ::ARD{T,K}, x::Array{T}, y::Array{T}) = kappa(κ.kernel, scprod(x, y, κ.weights))

kernel_dx{T<:FloatingPoint,K<:SquaredDistanceKernel}(κ::ARD{T,K}, x::Array{T}, y::Array{T}) = kappa_dz(κ.kernel, sqdist(x, y, κ.weights)) * sqdist_dx(x, y, κ.weights)
kernel_dy{T<:FloatingPoint,K<:StandardKernel}(κ::ARD{T,K}, x::Array{T}, y::Array{T}) = kappa_dz(κ.kernel, sqdist(x, y, κ.weights)) * sqdist_dy(x, y, κ.weights)


