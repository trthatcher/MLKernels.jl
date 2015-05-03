function kernelmatrix_dk_dx{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T},
                                        trans::Char = 'N')
    idx = trans == 'N' ? 1 : 2
    n = size(X, idx)
    m = size(Y, idx)
    idx = trans == 'N' ? 2 : 1
    (d = size(X, idx)) == size(Y, idx) || throw(ArgumentError(
            "X and Y do not have the same number of " * trans == 'N' ? "rows." : "columns."))
    K = Array(T, d, n, m)
    if trans == 'N'
        for j = 1:m 
            for i = 1:n
                K[:,i,j] = kernel_dx(κ, X[i,:], Y[j,:])
            end
        end
    else
        for j = 1:m 
            for i = 1:n
                K[:,i,j] = kernel_dx(κ, X[:,i], Y[:,j])
            end
        end
    end
    reshape(K, (d*n, m))
end

function kernelmatrix_dk_dy{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T},
                                        trans::Char = 'N')
    idx = trans == 'N' ? 1 : 2
    n = size(X, idx)
    m = size(Y, idx)
    idx = trans == 'N' ? 2 : 1
    (d = size(X, idx)) == size(Y, idx) || throw(ArgumentError(
            "X and Y do not have the same number of " * trans == 'N' ? "rows." : "columns."))
    K = Array(T, d, n, m)
    if trans == 'N'
        for j = 1:m 
            for i = 1:n
                K[:,i,j] = kernel_dy(κ, X[i,:], Y[j,:])
            end
        end
    else
        for j = 1:m 
            for i = 1:n
                K[:,i,j] = kernel_dy(κ, X[:,i], Y[:,j])
            end
        end
    end
    reshape(permutedims(K, [2,1,3]), (n, d*m))
end

function kernelmatrix_d2k_dxdy{T<:FloatingPoint}(κ::StandardKernel{T}, X::Matrix{T}, Y::Matrix{T},
                                        trans::Char = 'N')
    idx = trans == 'N' ? 1 : 2
    n = size(X, idx)
    m = size(Y, idx)
    idx = trans == 'N' ? 2 : 1
    (d = size(X, idx)) == size(Y, idx) || throw(ArgumentError(
            "X and Y do not have the same number of " * trans == 'N' ? "rows." : "columns."))
    K = Array(T, d, n, d, m)
    if trans == 'N'
        @inbounds for j = 1:m 
            for i = 1:n
                #K[:,:,i,j] = kernel_dxdy(κ, X[i,:], Y[j,:])
                kernel_dxdy!(κ, d, K, i, j, X, Y)
            end
        end
    else
        warn("Not implemented properly")
        @inbounds for j = 1:m 
            for i = 1:n
                #K[:,:,i,j] = kernel_dxdy(κ, X[:,i], Y[:,j])
                kernel_dxdy!(κ, K, i, j, X[:,i], Y[:,j])
            end
        end
    end
    reshape(K, (d*n, d*m))
end

function kernelmatrix_d2k_dxdy{T<:FloatingPoint}(κ::StandardKernel{T}, X::Vector{T}, Y::Vector{T})
    n = length(X)
    m = length(Y)
    K = Array(T, n, m)
    @inbounds for j = 1:m 
        for i = 1:n
            K[i,j] = kernel_dxdy(κ, X[i], Y[j])
        end
    end
    K
end


