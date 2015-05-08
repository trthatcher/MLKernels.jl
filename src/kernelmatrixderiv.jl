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
    @transpose_access is_trans (X,Y) @inbounds for j = 1:m 
            for i = 1:n
                K[:,i,j] = kernel_dx(κ, X[i,:], Y[j,:])
            end
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
    @transpose_access is_trans (X,Y) @inbounds for j = 1:m 
            for i = 1:n
                K[i,:,j] = kernel_dy(κ, X[i,:], Y[j,:])
            end
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
    if trans == 'N'
        @inbounds for j = 1:m, i = 1:n
            N_kernel_dxdy!(κ, d, K, X, i, Y, j)
        end
    else
        @inbounds for j = 1:m, i = 1:n
            T_kernel_dxdy!(κ, d, K, X, i, Y, j)
        end
    end
    K # reshape(K, (d*n, d*m))
end

#==========================================================================
  Optimized Kernel Derivative Matrices for Squared Distance Kernels
==========================================================================#

for (TN_kernel_dxdy!, TN_sqdist, is_T) in ((:N_kernel_dxdy!, :N_sqdist, false), 
                                           (:T_kernel_dxdy!, :T_sqdist, true))
    @eval begin

        # Squared distance between vectors X[x_pos,:] and Y[y_pos,:] (X[:,x_pos] and Y[:,y_pos])
        function $TN_sqdist{T<:FloatingPoint}(d::Int64, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64)
            z = zero(T)
            @inbounds @simd for i = 1:d
                v = $(is_T ? :(X[i,x_pos] - Y[i,y_pos]) : :(X[x_pos,i] - Y[y_pos,i]))
                z += v*v
            end
            z
        end

        # In-place update of K[x_pos,:,y_pos,:] (K[:,x_pos,:,y_pos]) to contain ∂k²/∂x∂y
        function $TN_kernel_dxdy!{T<:FloatingPoint}(κ::SquaredDistanceKernel{T}, d::Int64, K::Array{T}, X::Array{T}, x_pos::Int64, Y::Array{T}, y_pos::Int64)
            ϵᵀϵ = $TN_sqdist(d, X, x_pos, Y, y_pos)
            ∂κ∂z = kappa_dz(κ, ϵᵀϵ)
            ∂κ²∂z² = kappa_dz2(κ, ϵᵀϵ)
            @inbounds for j = 1:d
                v = $(is_T ? :(X[j,x_pos] - Y[j,y_pos]) : :(X[x_pos,j] - Y[y_pos,j]))
                for i = 1:d
                    K[i,x_pos,j,y_pos] = -4∂κ²∂z² * v *  $(is_T ? :(X[i,x_pos] - Y[i,y_pos]) : :(X[x_pos,i] - Y[y_pos,i]))
                end
                K[j,x_pos,j,y_pos] -= 2∂κ∂z
            end
            K
        end

    end
end


