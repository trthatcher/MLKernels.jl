#===================================================================================================
  Standard Kernel Derivatives
===================================================================================================#

# concrete kernel types should provide kernel_dp(::<KernelType>, param::Symbol, x, y)
kernel_dp{T<:FloatingPoint}(κ::StandardKernel{T}, param::Integer, x::Array{T}, y::Array{T}) = kernel_dp(κ, kernelparameters(κ)[param], x, y)
kernel_dp{T<:FloatingPoint}(κ::StandardKernel{T}, param::Integer, x::T, y::T) = kernel_dp(κ, kernelparameters(κ)[param], x, y)


#===========================================================================
  Scalar Product Kernel Derivatives
===========================================================================#

function kernel_dx{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::Array{T}, y::Array{T}) # = kappa_dz(κ, scprod(x, y)) * scprod_dx(x, y)
    ∂κ_∂z = kappa_dz(κ, scprod(x, y))
    d = length(x)
    ∂k_∂x = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂x[i] = ∂κ_∂z * y[i]
    end
    ∂k_∂x
end
kernel_dy{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::Array{T}, y::Array{T}) = kernel_dx(κ, y, x)

function kernel_dxdy{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::Array{T}, y::Array{T})
    xᵀy = scprod(x, y)
    ∂κ_∂z = kappa_dz(κ, xᵀy)
    ∂κ²_∂z² = kappa_dz2(κ, xᵀy)
    d = length(x)
    ∂k²_∂x∂y = Array(T, d, d)
    @inbounds for j = 1:d
        for i = 1:d
            ∂k²_∂x∂y[i,j] = ∂κ²_∂z² * y[i] * x[j]
        end
        ∂k²_∂x∂y[j,j] += ∂κ_∂z
    end
    ∂k²_∂x∂y
end

kernel_dx{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::T, y::T) = kappa_dz(κ, x*y) * y
kernel_dy{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::T, y::T) = kappa_dz(κ, x*y) * x

function kernel_dxdy{T<:FloatingPoint}(κ::ScalarProductKernel{T}, x::T, y::T)
    xy = x * y
    kappa_dz2(κ, xy) * xy + kappa_dz(κ, xy)
end

kernel_dp{T<:FloatingPoint}(κ::ScalarProductKernel{T}, param::Symbol, x::Array{T}, y::Array{T}) = kappa_dp(κ, param, scprod(x, y))
kernel_dp{T<:FloatingPoint}(κ::ScalarProductKernel{T}, param::Symbol, x::T, y::T) = kappa_dp(κ, param, x*y)


#===========================================================================
  Squared Distance Kernel Derivatives
===========================================================================#

function kernel_dx{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::Array{T}, y::Array{T}) # = kappa_dz(κ, sqdist(x, y)) * sqdist_dx(x, y)
    ∂κ_∂z = kappa_dz(κ, sqdist(x, y))
    d = length(x)
    ∂k_∂x = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂x[i] = 2∂κ_∂z * (x[i] - y[i])
    end
    ∂k_∂x
end
kernel_dy{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::Array{T}, y::Array{T}) = kernel_dx(κ, y, x)

kernel_dp{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, param::Symbol, x::Array{T}, y::Array{T}) = kappa_dp(κ, param, sqdist(x, y))
kernel_dp{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, param::Symbol, x::T, y::T) = kappa_dp(κ, param, (x-y)^2)

function kernel_dxdy{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::Array{T}, y::Array{T})
    ϵᵀϵ = sqdist(x, y)
    ∂κ_∂z = kappa_dz(κ, ϵᵀϵ)
    ∂κ²_∂z² = kappa_dz2(κ, ϵᵀϵ)
    d = length(x)
    ∂k²_∂x∂y = Array(T, d, d)
    @inbounds for j = 1:d
        ϵj = x[j] - y[j]
        for i = 1:d
            ϵi = x[i] - y[i]
            ∂k²_∂x∂y[i,j] = -4∂κ²_∂z² * ϵj * ϵi
        end
        ∂k²_∂x∂y[j,j] -= 2∂κ_∂z
    end
    ∂k²_∂x∂y
end

kernel_dx{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::T, y::T) = kappa_dz(κ, (x-y)^2) * 2(x-y)
kernel_dy{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::T, y::T) = kappa_dz(κ, (x-y)^2) * 2(y-x)

function kernel_dxdy{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, x::T, y::T)
    ϵᵀϵ = (x-y)^2
    -kappa_dz2(κ, ϵᵀϵ) * 4ϵᵀϵ - 2kappa_dz(κ, ϵᵀϵ)
end


#===========================================================================
  Separable Kernel Derivatives
===========================================================================#

function kappa_dz_array!{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T})
    @inbounds for i = 1:length(x)
        x[i] = kappa_dz(κ, x[i])
    end
    x
end

function kernel_dx{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T}, y::Array{T})
    v = kappa_dz_array!(κ, copy(x))
    z = kappa_array!(κ, copy(y))
    v.*z
end

function kernel_dy{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T}, y::Array{T})
    v = kappa_array!(κ, copy(x))
    z = kappa_dz_array!(κ, copy(y))
    v.*z
end

function kernel_dxdy{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T}, y::Array{T})
    v = kappa_dz_array!(κ, copy(x))
    z = kappa_dz_array!(κ, copy(y))
    diagm(v.*z)
end

kernel_dx{T<:FloatingPoint}(κ::SeparableKernel{T}, x::T, y::T) = kappa_dz(κ, x) * kappa(κ, y)
kernel_dy{T<:FloatingPoint}(κ::SeparableKernel{T}, x::T, y::T) = kappa(κ, x) * kappa_dz(κ, y)
kernel_dxdy{T<:FloatingPoint}(κ::SeparableKernel{T}, x::T, y::T) = kappa_dz(κ, x) * kappa_dz(κ, y)

function kernel_dp{T<:FloatingPoint}(κ::SeparableKernel{T}, param::Symbol, x::Array{T}, y::Array{T})
    (n = length(x)) == length(y) || error("Dimensions do not match")
    v = zero(T)
    @inbounds for i = 1:n
        v += kappa_dp(κ, param, x[i])*kappa(κ, y[i]) + kappa(κ, x[i])*kappa_dp(κ, param, y[i])
    end
    v
end
kernel_dp{T<:FloatingPoint}(κ::SeparableKernel{T}, param::Symbol, x::T, y::T) = kappa_dp(κ, param, x)*kappa(κ, y) + kappa(κ, x)*kappa_dp(κ, param, y)


#===================================================================================================
  Automatic Relevance Determination (ARD) Kernel Derivatives
===================================================================================================#

#=== ARD Squared Distance ===#

function kernel_dx{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    ∂κ_∂z = kappa_dz(κ.k, sqdist(x, y, w))
    d = length(x)
    ∂k_∂x = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂x[i] = 2∂κ_∂z * (x[i] - y[i]) * w[i]^2
    end
    ∂k_∂x
end
kernel_dy{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kernel_dx(κ, y, x)

function kernel_dweights{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    ∂κ_∂z = kappa_dz(κ.k, sqdist(x, y, w))
    d = length(x)
    ∂k_∂w = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂w[i] = 2∂κ_∂z * (x[i] - y[i])^2 * w[i]
    end
    ∂k_∂w
end

function kernel_dxdy{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    ϵᵀW²ϵ = sqdist(x, y, w)
    ∂κ_∂z = kappa_dz(κ.k, ϵᵀW²ϵ)
    ∂κ²_∂z² = kappa_dz2(κ.k, ϵᵀW²ϵ)
    d = length(x)
    ∂k²_∂x∂y = Array(T, d, d)
    @inbounds for j = 1:d
        wj² = w[j]^2
        ϵj = (x[j] - y[j]) * wj²
        for i = 1:d
            ϵi = (x[i] - y[i]) * w[i]^2
            ∂k²_∂x∂y[i,j] = -4∂κ²_∂z² * ϵj * ϵi
        end
        ∂k²_∂x∂y[j,j] -= 2∂κ_∂z * wj²
    end
    ∂k²_∂x∂y
end

function kernel_dp{T<:FloatingPoint,U<:SquaredDistanceKernel}(κ::ARD{T,U}, param::Symbol, x::Array{T}, y::Array{T})
    if param == :weights
        return kernel_dweights(κ, x, y)
    else
        return kappa_dp(κ.k, param, sqdist(x, y, κ.weights))
    end
end

#=== ARD Scalar Product ===#

function kernel_dx{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    ∂κ_∂z = kappa_dz(κ.k, scprod(x, y, w))
    d = length(x)
    ∂k_∂x = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂x[i] = ∂κ_∂z * y[i] * w[i]^2
    end
    ∂k_∂x
end
kernel_dy{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T}) = kernel_dx(κ, y, x)

function kernel_dweights{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    ∂κ_∂z = kappa_dz(κ.k, scprod(x, y, w))
    d = length(x)
    ∂k_∂w = Array(T, d)
    @inbounds @simd for i = 1:d
        ∂k_∂w[i] = 2∂κ_∂z * x[i] * y[i] * w[i]
    end
    ∂k_∂w
end

function kernel_dp{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, param::Symbol, x::Array{T}, y::Array{T})
    if param == :weights
        return kernel_dweights(κ, x, y)
    else
        return kappa_dp(κ.k, param, scprod(x, y, κ.weights))
    end
end



function kernel_dxdy{T<:FloatingPoint,U<:ScalarProductKernel}(κ::ARD{T,U}, x::Array{T}, y::Array{T})
    w = κ.weights
    xᵀW²y = scprod(x, y, w)
    ∂κ_∂z = kappa_dz(κ.k, xᵀW²y)
    ∂κ²_∂z² = kappa_dz2(κ.k, xᵀW²y)
    d = length(x)
    ∂k²_∂x∂y = Array(T, d, d)
    @inbounds for j = 1:d
        wj² = w[j]^2
        v = ∂κ²_∂z² * x[j] * wj²
        for i = 1:d
            ∂k²_∂x∂y[i,j] = v * w[i]^2 * y[i]
        end
        ∂k²_∂x∂y[j,j] += ∂κ_∂z * wj²
    end
    ∂k²_∂x∂y
end



#===================================================================================================
  Composite Kernels
===================================================================================================#

#===========================================================================
  Product Kernel Derivatives
===========================================================================#

function kernel_dx{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T})
    ks = [kernel(ψ.k[i], x, y) for i=1:length(ψ.k)]
    ψ.a * prod(ks) * sum([kernel_dx(ψ.k[i], x, y)/ks[i] for i=1:length(ψ.k)])
end

function kernel_dy{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T})
    ks = [kernel(ψ.k[i], x, y) for i=1:length(ψ.k)]
    ψ.a * prod(ks) * sum([kernel_dy(ψ.k[i], x, y)/ks[i] for i=1:length(ψ.k)])
end

#function kernel_dxdy{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T})
#    ψ.a * (kernel_dxdy(ψ.k1, x, y)*kernel(ψ.k2, x, y)
#            + kernel_dy(ψ.k1, x, y)*kernel_dx(ψ.k2, x, y)'
#            + kernel_dx(ψ.k1, x, y)*kernel_dy(ψ.k2, x, y)'
#            + kernel(ψ.k1, x, y)*kernel_dxdy(ψ.k2, x, y))
#end

#function kernel_dp{T<:FloatingPoint}(ψ::KernelProduct{T}, param::Symbol, x::Vector{T}, y::Vector{T})
#    if param == :a
#        kernel(ψ.k1, x, y) * kernel(ψ.k2, x, y)
#    elseif (sparam = string(param); beginswith(sparam, "k1."))
#        subparam = symbol(sparam[4:end])
#        ψ.a * kernel_dp(ψ.k1, subparam, x, y) * kernel(ψ.k2, x, y)
#    elseif beginswith(sparam, "k2.")
#        subparam = symbol(sparam[4:end])
#        ψ.a * kernel(ψ.k1, x, y) * kernel_dp(ψ.k2, subparam, x, y)
#    else
#        warn("derivative with respect to unrecognized symbol")
#        zero(T)
#    end
#end
#
#function kernel_dp{T<:FloatingPoint}(ψ::KernelProduct{T}, param::Integer, x::Vector{T}, y::Vector{T})
#    N1 = length(kernelparameters(ψ.k1))
#    N2 = length(kernelparameters(ψ.k2))
#    if param == 1
#        kernel_dp(ψ, :a, x, y)
#    elseif 2 <= param <= N1 + 1
#        ψ.a * kernel_dp(ψ.k1, param-1, x, y) * kernel(ψ.k2, x, y)
#    elseif N1 + 2 <= param <= N1 + N2 + 1
#        ψ.a * kernel(ψ.k1, x, y) * kernel_dp(ψ.k2, param-N1-1, x, y)
#    else
#        throw(ArgumentError("param must be between 1 and $(N1+N2+1)"))
#    end
#end


#===========================================================================
  Kernel Sum
===========================================================================#

function kernel_dx{T<:FloatingPoint}(ψ::KernelSum{T}, x::Vector{T}, y::Vector{T})
    sum([kernel_dx(ψ.k[i], x, y) for i=1:length(ψ.k)])
end

function kernel_dy{T<:FloatingPoint}(ψ::KernelSum{T}, x::Vector{T}, y::Vector{T})
    sum([kernel_dy(ψ.k[i], x, y) for i=1:length(ψ.k)])
end

function kernel_dxdy{T<:FloatingPoint}(ψ::KernelSum{T}, x::Vector{T}, y::Vector{T})
    sum([kernel_dxdy(ψ.k[i], x, y) for i=1:length(ψ.k)])
end

#function kernel_dp{T<:FloatingPoint}(ψ::KernelSum{T}, param::Symbol, x::Vector{T}, y::Vector{T})
#    if param == :a1
#        kernel(ψ.k1, x, y)
#    elseif param == :a2
#        kernel(ψ.k2, x, y)
#    elseif (sparam = string(param); beginswith(sparam, "k1."))
#        subparam = symbol(sparam[4:end])
#        ψ.a1 * kernel_dp(ψ.k1, subparam, x, y)
#    elseif beginswith(sparam, "k2.")
#        subparam = symbol(sparam[4:end])
#        ψ.a2 * kernel_dp(ψ.k2, subparam, x, y)
#    else
#        warn("derivative with respect to unrecognized symbol")
#        zero(T)
#    end
#end
#
#function kernel_dp{T<:FloatingPoint}(ψ::KernelSum{T}, param::Integer, x::Vector{T}, y::Vector{T})
#    N1 = length(kernelparameters(ψ.k1))
#    N2 = length(kernelparameters(ψ.k2))
#    if param == 1
#        kernel_dp(ψ, :a1, x, y)
#    elseif 2 <= param <= N1 + 1
#        ψ.a1 * kernel_dp(ψ.k1, param-1, x, y)
#    elseif param == N1 + 2
#        kernel_dp(ψ, :a2, x, y)
#    elseif N1 + 3 <= param <= N1 + N2 + 2
#        ψ.a2 * kernel_dp(ψ.k2, param-N1-2, x, y)
#    else
#        throw(ArgumentError("param must be between 1 and $(N1+N2+2)"))
#    end
#end
