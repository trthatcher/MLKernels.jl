#===================================================================================================
  Standard Kernel Derivatives
===================================================================================================#

# concrete kernel types should provide kernel_dp(::<KernelType>, param::Symbol, x, y)
kernel_dp{T<:FloatingPoint}(κ::StandardKernel{T}, param::Integer, x::KernelInput{T}, y::KernelInput{T}) = kernel_dp(κ, kernelparameters(κ)[param], x, y)


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

kernel_dp{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, param::Symbol, x::Array{T}, y::Array{T}) = kappa_dp(κ, param, sqdist(x, y))
kernel_dp{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, param::Symbol, x::T, y::T) = kappa_dp(κ, param, (x-y)^2)


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
    v .* z
end

function kernel_dy{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T}, y::Array{T})
    v = kappa_array!(κ, copy(x))
    z = kappa_dz_array!(κ, copy(y))
    v .* z
end

function kernel_dxdy{T<:FloatingPoint}(κ::SeparableKernel{T}, x::Array{T}, y::Array{T})
    v = kappa_dz_array!(κ, copy(x))
    z = kappa_dz_array!(κ, copy(y))
    diagm(v .* z)
end

kernel_dx{T<:FloatingPoint}(κ::SeparableKernel{T}, x::T, y::T) = kappa_dz(κ, x) * kappa(κ, y)
kernel_dy{T<:FloatingPoint}(κ::SeparableKernel{T}, x::T, y::T) = kappa(κ, x) * kappa_dz(κ, y)
kernel_dxdy{T<:FloatingPoint}(κ::SeparableKernel{T}, x::T, y::T) = kappa_dz(κ, x) * kappa_dz(κ, y)

function kernel_dp{T<:FloatingPoint}(κ::SeparableKernel{T}, param::Symbol, x::Array{T}, y::Array{T})
    (n = length(x)) == length(y) || error("Dimensions do not match")
    v = zero(T)
    @inbounds for i = 1:n
        v += kappa_dp(κ, param, x[i]) * kappa(κ, y[i]) + kappa(κ, x[i]) * kappa_dp(κ, param, y[i])
    end
    v
end
kernel_dp{T<:FloatingPoint}(κ::SeparableKernel{T}, param::Symbol, x::T, y::T) = kappa_dp(κ, param, x) * kappa(κ, y) + kappa(κ, x) * kappa_dp(κ, param, y)


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

function kernel_dx{T<:FloatingPoint}(ψ::KernelProduct{T}, x::KernelInput{T}, y::KernelInput{T})
    ks = map(κ -> kernel(κ,x,y), ψ.k)
    ψ.a * prod(ks) * sum([kernel_dx(ψ.k[i], x, y) / ks[i] for i=1:length(ψ.k)])
end

function kernel_dy{T<:FloatingPoint}(ψ::KernelProduct{T}, x::KernelInput{T}, y::KernelInput{T})
    ks = map(κ -> kernel(κ,x,y), ψ.k)
    ψ.a * prod(ks) * sum([kernel_dy(ψ.k[i], x, y) / ks[i] for i=1:length(ψ.k)])
end

function kernel_dxdy{T<:FloatingPoint}(ψ::KernelProduct{T}, x::Vector{T}, y::Vector{T})
    n = length(ψ.k)
    ks = map(κ -> kernel(κ,x,y), ψ.k)
    k_dx = [kernel_dx(ψ.k[i], x, y) for i=1:n]
    k_dy = [kernel_dy(ψ.k[i], x, y) for i=1:n]
    dxdy = zeros(T, length(x), length(y))
    for i = 1:n
        prod_ks_1_i1 = prod(ks[1:i-1])
        dxdy += prod_ks_1_i1 * kernel_dxdy(ψ.k[i], x, y) * prod(ks[i+1:end])
        for j = i+1:n
            dxdy += prod_ks_1_i1 * prod(ks[i+1:j-1]) * prod(ks[j+1:end]) * (k_dx[j]*k_dy[i]' + k_dx[i]*k_dy[j]')
        end
    end
    ψ.a * dxdy
end

function kernel_dxdy{T<:FloatingPoint}(ψ::KernelProduct{T}, x::T, y::T)
    n = length(ψ.k)
    ks = map(κ -> kernel(κ,x,y), ψ.k)
    k_dx = map(κ -> kernel_dx(κ,x,y), ψ.k)
    k_dy = map(κ -> kernel_dy(κ,x,y), ψ.k)
    dxdy = zero(T)
    for i = 1:n
        prod_ks_1_i1 = prod(ks[1:i-1])
        dxdy += prod_ks_1_i1 * kernel_dxdy(ψ.k[i], x, y) * prod(ks[i+1:end])
        for j = i+1:n
            dxdy += prod_ks_1_i1 * prod(ks[i+1:j-1]) * prod(ks[j+1:end]) * (k_dy[i]*k_dx[j] + k_dx[i]*k_dy[j])
        end
    end
    ψ.a * dxdy
end

function kernel_dp{T<:FloatingPoint}(ψ::KernelProduct{T}, param::Symbol, x::KernelInput{T}, y::KernelInput{T})
    if param == :a
        prod(map(κ -> kernel(κ,x,y), ψ.k))
    elseif (idx = indexin([param], kernelparameters(ψ))[1]) != 0
        kernel_dp(ψ, idx, x, y)
    else
        warn("derivative with respect to unrecognized symbol")
        zero(T)
    end
end

function kernel_dp{T<:FloatingPoint}(ψ::KernelProduct{T}, param::Integer, x::KernelInput{T}, y::KernelInput{T})
    n = length(ψ.k)
    nps = map(κ -> length(kernelparameters(κ)), ψ.k)
    totN = 1 + sum(nps)
    if param == 1
        kernel_dp(ψ, :a, x, y)
    elseif 2 <= param <= totN
        i = 1
        p = param - 1
        while i < n && p > nps[i]
            p -= nps[i]
            i += 1
        end

        ks = map(κ -> kernel(κ,x,y), ψ.k)
        ψ.a * prod(ks[1:i-1]) * kernel_dp(ψ.k[i], p, x, y) * prod(ks[i+1:end])
    else
        throw(ArgumentError("param must be between 1 and $(totN)"))
    end
end


#===========================================================================
  Kernel Sum
===========================================================================#

function kernel_dx{T<:FloatingPoint}(ψ::KernelSum{T}, x::KernelInput{T}, y::KernelInput{T})
    sum(map(κ -> kernel_dx(κ, x, y), ψ.k))
end

function kernel_dy{T<:FloatingPoint}(ψ::KernelSum{T}, x::KernelInput{T}, y::KernelInput{T})
    sum(map(κ -> kernel_dy(κ, x, y), ψ.k))
end

function kernel_dxdy{T<:FloatingPoint}(ψ::KernelSum{T}, x::KernelInput{T}, y::KernelInput{T})
    sum(map(κ -> kernel_dxdy(κ, x, y), ψ.k))
end

function kernel_dp{T<:FloatingPoint}(ψ::KernelSum{T}, param::Symbol, x::KernelInput{T}, y::KernelInput{T})
    if (idx = indexin([param], kernelparameters(ψ))[1]) != 0
        kernel_dp(ψ, idx, x, y)
    else
        warn("derivative with respect to unrecognized symbol")
        zero(T)
    end
end

function kernel_dp{T<:FloatingPoint}(ψ::KernelSum{T}, param::Integer, x::KernelInput{T}, y::KernelInput{T})
    n = length(ψ.k)
    nps = map(κ -> length(kernelparameters(κ)), ψ.k)
    totN = sum(nps)
    if 1 <= param <= totN
        i = 1
        p = param
        while i < n && p > nps[i]
            p -= nps[i]
            i += 1
        end

        kernel_dp(ψ.k[i], p, x, y)
    else
        throw(ArgumentError("param must be between 1 and $(totN)"))
    end
end
