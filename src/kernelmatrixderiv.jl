#===================================================================================================
  Kernel Derivative Matrices
===================================================================================================#

#==========================================================================
  Generic Kernel Derivative Matrices
==========================================================================#

immutable Derivative{T<:FloatingPoint} <: SimpleKernel{T}
    kernel::Kernel{T}
    deriv::Union(Symbol,Integer)
end

kernel{T<:FloatingPoint}(κ::Derivative{T}, x::KernelInput{T}, y::KernelInput{T}) = kernel_dp(κ.kernel, κ.deriv, x, y)

kernelmatrix_dp{T<:FloatingPoint}(κ::Kernel{T}, deriv::Union(Symbol,Integer), X::Matrix{T}, trans::Char = 'N', uplo::Char = 'U', sym::Bool = true) = kernelmatrix(Derivative(κ, deriv), X, trans, uplo, sym)
kernelmatrix_dp{T<:FloatingPoint}(κ::Kernel{T}, deriv::Union(Symbol,Integer), X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N') = kernelmatrix(Derivative(κ, deriv), X, Y, trans)

function kernel_dx!{T<:FloatingPoint}(κ::Kernel{T}, d::Int64, K::Array{T}, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    if is_trans
        K[1:d,x_pos,y_pos] = kernel_dx(κ, vec(X[1:d,x_pos]), vec(Y[1:d,y_pos]))
    else
        K[1:d,x_pos,y_pos] = kernel_dx(κ, vec(X[x_pos,1:d]), vec(Y[y_pos,1:d]))
    end
end

function generic_kernelmatrix_dx{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    is_trans = trans == 'T'  # True if columns are observations
    n = size(X, is_trans ? 2 : 1)
    m = size(Y, is_trans ? 2 : 1)
    if (d = size(X, is_trans ? 1 : 2)) != size(Y, is_trans ? 1 : 2)
        throw(ArgumentError("X and Y do not have the same number of " * (is_trans ? "rows." : "columns.")))
    end
    K = Array(T, d, n, m)
    @inbounds for j = 1:m, i = 1:n
        K[:,i,j] = kernel_dx!(κ, d, K, X, i, Y, j, is_trans)
    end
    reshape(K, (d*n, m))
end

kernelmatrix_dx{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N') = generic_kernelmatrix_dx(κ, X, Y, trans)


function kernel_dy!{T<:FloatingPoint}(κ::Kernel{T}, d::Int64, K::Array{T}, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    if is_trans
        K[x_pos,1:d,y_pos] = kernel_dy(κ, vec(X[1:d,x_pos]), vec(Y[1:d,y_pos]))
    else
        K[x_pos,1:d,y_pos] = kernel_dy(κ, vec(X[x_pos,1:d]), vec(Y[y_pos,1:d]))
    end
end

function generic_kernelmatrix_dy{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    is_trans = trans == 'T'  # True if columns are observations
    n = size(X, is_trans ? 2 : 1)
    m = size(Y, is_trans ? 2 : 1)
    if (d = size(X, is_trans ? 1 : 2)) != size(Y, is_trans ? 1 : 2)
        throw(ArgumentError("X and Y do not have the same number of " * (is_trans ? "rows." : "columns.")))
    end
    K = Array(T, n, d, m)
    @inbounds for j = 1:m, i = 1:n
        K[i,:,j] = kernel_dy!(κ, d, K, X, i, Y, j, is_trans)
    end
    reshape(K, (n, d*m))
end

kernelmatrix_dy{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N') = generic_kernelmatrix_dy(κ, X, Y, trans)

function kernel_dxdy!{T<:FloatingPoint}(κ::Kernel{T}, d::Int64, K::Array{T}, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    if is_trans
        K[1:d,x_pos,1:d,y_pos] = kernel_dxdy(κ, vec(X[1:d,x_pos]), vec(Y[1:d,y_pos]))
    else
        K[1:d,x_pos,1:d,y_pos] = kernel_dxdy(κ, vec(X[x_pos,1:d]), vec(Y[y_pos,1:d]))
    end
end

function generic_kernelmatrix_dxdy{T<:FloatingPoint}(κ::Kernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    is_trans = trans == 'T'  # True if columns are observations
    n = size(X, is_trans ? 2 : 1)
    m = size(Y, is_trans ? 2 : 1)
    if (d = size(X, is_trans ? 1 : 2)) != size(Y, is_trans ? 1 : 2)
        throw(ArgumentError("X and Y do not have the same number of " * (is_trans ? "rows." : "columns.")))
    end
    K = Array(T, d, n, d, m)
    @inbounds for j = 1:m, i = 1:n
        kernel_dxdy!(κ, d, K, X, i, Y, j, is_trans)
    end
    K # reshape(K, (d*n, d*m))
end

kernelmatrix_dxdy{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N') = generic_kernelmatrix_dxdy(κ, X, Y, trans)


#==========================================================================
  Optimized Kernel Derivative Matrices for Scalar Product Kernels
==========================================================================#

function kernel_dxdy!{T<:FloatingPoint}(κ::ScalarProductKernel{T}, d::Int64, K::Array{T}, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    xᵀy = scprod(d, X, x_pos, Y, y_pos, is_trans)
    ∂κ_∂z = kappa_dz(κ, xᵀy)
    ∂κ²_∂z² = kappa_dz2(κ, xᵀy)
    @transpose_access is_trans (X,Y) @inbounds for j = 1:d 
        for i = 1:d
            K[j,x_pos,i,y_pos] = ∂κ²_∂z² * X[x_pos,i] * Y[y_pos,j]
        end
        K[j,x_pos,j,y_pos] += ∂κ_∂z
    end
    K
end


#==========================================================================
  Optimized Kernel Derivative Matrices for Squared Distance Kernels
==========================================================================#

function kernel_dxdy!{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, d::Int64, K::Array{T}, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    ϵᵀϵ = sqdist(d, X, x_pos, Y, y_pos, is_trans)
    ∂κ_∂z = kappa_dz(κ, ϵᵀϵ)
    ∂κ²_∂z² = kappa_dz2(κ, ϵᵀϵ)
    @transpose_access is_trans (X,Y) @inbounds for j = 1:d
        v = X[x_pos,j] - Y[y_pos,j]
        for i = 1:d
            K[i,x_pos,j,y_pos] = -4∂κ²_∂z² * v * (X[x_pos,i] - Y[y_pos,i])
        end
        K[j,x_pos,j,y_pos] -= 2∂κ_∂z
    end
    K
end


#==========================================================================
  Kernel Derivative Matrices for Composite Kernels
==========================================================================#

function kernelmatrix_dx{T<:FloatingPoint}(κ::KernelSum{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    c = length(κ.k)
    K = kernelmatrix_dx(κ.k[1], X, Y, trans)
    if c > 1
        for i = 2:c
            BLAS.axpy!(one(T), kernelmatrix_dx(κ.k[i], X, Y, trans), K)
        end
    end
    K
end

function kernelmatrix_dy{T<:FloatingPoint}(κ::KernelSum{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    c = length(κ.k)
    K = kernelmatrix_dy(κ.k[1], X, Y, trans)
    for i = 2:c
        BLAS.axpy!(one(T), kernelmatrix_dy(κ.k[i], X, Y, trans), K)
    end
    K
end

function kernelmatrix_dxdy{T<:FloatingPoint}(κ::KernelSum{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    c = length(κ.k)
    K = kernelmatrix_dxdy(κ.k[1], X, Y, trans)
    for i = 2:c
        BLAS.axpy!(one(T), kernelmatrix_dxdy(κ.k[i], X, Y, trans), K)
    end
    K
end

kernelmatrix_dx{T<:FloatingPoint}(κ::KernelProduct{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N') = generic_kernelmatrix_dx(κ, X, Y, trans)
kernelmatrix_dy{T<:FloatingPoint}(κ::KernelProduct{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N') = generic_kernelmatrix_dy(κ, X, Y, trans)
kernelmatrix_dxdy{T<:FloatingPoint}(κ::KernelProduct{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N') = generic_kernelmatrix_dxdy(κ, X, Y, trans)

kernelmatrix_dx(k::ARD, X::Matrix, Y::Matrix, trans::Char = 'N') = generic_kernelmatrix_dx(k, X, Y, trans)
kernelmatrix_dy(k::ARD, X::Matrix, Y::Matrix, trans::Char = 'N') = generic_kernelmatrix_dy(k, X, Y, trans)
kernelmatrix_dxdy(k::ARD, X::Matrix, Y::Matrix, trans::Char = 'N') = generic_kernelmatrix_dxdy(k, X, Y, trans)
