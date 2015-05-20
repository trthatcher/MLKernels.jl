#===================================================================================================
  Kernel Derivative Matrices
===================================================================================================#

#==========================================================================
  Generic Kernel Derivative Matrices
==========================================================================#

function kernel_dx!{T<:FloatingPoint}(κ::StandardKernel{T}, d::Int64, K::Array{T}, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    if is_trans
        K[1:d,x_pos,y_pos] = kernel_dx(κ, vec(X[1:d,x_pos]), vec(Y[1:d,y_pos]))
    else
        K[1:d,x_pos,y_pos] = kernel_dx(κ, vec(X[x_pos,1:d]), vec(Y[y_pos,1:d]))
    end
end

function kernelmatrix_dx{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
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


function kernel_dy!{T<:FloatingPoint}(κ::StandardKernel{T}, d::Int64, K::Array{T}, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    if is_trans
        K[x_pos,1:d,y_pos] = kernel_dy(κ, vec(X[1:d,x_pos]), vec(Y[1:d,y_pos]))
    else
        K[x_pos,1:d,y_pos] = kernel_dy(κ, vec(X[x_pos,1:d]), vec(Y[y_pos,1:d]))
    end
end

function kernelmatrix_dy{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
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


function kernel_dxdy!{T<:FloatingPoint}(κ::StandardKernel{T}, d::Int64, K::Array{T}, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    if is_trans
        K[1:d,x_pos,1:d,y_pos] = kernel_dxdy(κ, vec(X[1:d,x_pos]), vec(Y[1:d,y_pos]))
    else
        K[1:d,x_pos,1:d,y_pos] = kernel_dxdy(κ, vec(X[x_pos,1:d]), vec(Y[y_pos,1:d]))
    end
end

function kernelmatrix_dxdy{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
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
