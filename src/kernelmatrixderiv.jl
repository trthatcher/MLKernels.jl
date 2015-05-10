#===================================================================================================
  Kernel Derivative Matrices
===================================================================================================#

#==========================================================================
  Generic Kernel Derivative Matrices
==========================================================================#

function kernelmatrix_dx{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    is_trans = trans == 'T'  # True if columns are observations
    n = size(X, is_trans ? 2 : 1)
    m = size(Y, is_trans ? 2 : 1)
    if (d = size(X, is_trans ? 1 : 2)) != size(Y, is_trans ? 1 : 2)
        throw(ArgumentError("X and Y do not have the same number of " * (is_trans ? "rows." : "columns.")))
    end
    K = Array(T, d, n, m)
    @transpose_access is_trans (X,Y) @inbounds for j = 1:m, i = 1:n
        K[:,i,j] = kernel_dx(κ, X[i,:], Y[j,:])
    end
    reshape(K, (d*n, m))
end

function kernelmatrix_dy{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T}, trans::Char = 'N')
    is_trans = trans == 'T'  # True if columns are observations
    n = size(X, is_trans ? 2 : 1)
    m = size(Y, is_trans ? 2 : 1)
    if (d = size(X, is_trans ? 1 : 2)) != size(Y, is_trans ? 1 : 2)
        throw(ArgumentError("X and Y do not have the same number of " * (is_trans ? "rows." : "columns.")))
    end
    K = Array(T, n, d, m)
    @transpose_access is_trans (X,Y) @inbounds for j = 1:m, i = 1:n
        K[i,:,j] = kernel_dy(κ, X[i,:], Y[j,:])
    end
    reshape(K, (n, d*m))
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
  Optimized Kernel Derivative Matrices for Squared Distance Kernels
==========================================================================#

function kernel_dxdy!{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, d::Int64, K::Array{T}, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64, is_trans::Bool)
    ϵᵀϵ = sqdist(d, X, x_pos, Y, y_pos, is_trans)
    ∂κ∂z = kappa_dz(κ, ϵᵀϵ)
    ∂κ²∂z² = kappa_dz2(κ, ϵᵀϵ)
    @transpose_access is_trans (X,Y) @inbounds for j = 1:d
        v = X[x_pos,j] - Y[y_pos,j]
        for i = 1:d
            K[i,x_pos,j,y_pos] = -4∂κ²∂z² * v *  X[x_pos,i] - Y[y_pos,i]
        end
        K[j,x_pos,j,y_pos] -= 2∂κ∂z
    end
    K
end
