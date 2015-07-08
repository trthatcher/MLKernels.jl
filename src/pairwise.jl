

function pairwise!{T<:FloatingPoint}(K::Matrix{T}, κ::AdditiveKernel{T}, X::Matrix{T}, Y::Matrix{T})
    @inbounds for k = 1:m
        for j = 1:n
            v = 0
            @inbounds @simd for i = 1:p
                v += kappa(A[i,j], B[i,k])
            end
            C[j,k] = v
        end
     end 
end
