#==========================================================================
  Vector Operations
==========================================================================#

### Scalar Product Kernels

# Scalar Product of vectors x and y
function scprod{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    (n = length(x)) == length(y) || throw(ArgumentError("Dimensions do not conform."))
    c = zero(T)
    @inbounds @simd for i = 1:n
        c += x[i]*y[i]
    end
    c
end

# Weighted Scalar Product of x and y
function scprod{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    c = zero(T)
    @inbounds @simd for i = 1:n
        c += x[i]*y[i]*w[i]
    end
    c
end

dscprod_dx{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = copy(y)
dscprod_dy{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = copy(x)

function dscprod_dx!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    @inbounds @simd for i = 1:n
        x[i] = y[i]*w[i]
    end
    x
end

dscprod_dy!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = dscprod_dx!(y, x, w)
dscprod_dw!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = dscprod_dx!(w, x, y)

dscprod_dx{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = dscprod_dx!(similar(x), y, w)
dscprod_dy{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = dscprod_dy!(x, similar(y), w)
dscprod_dw{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = dscprod_dy!(x, y, similar(w))


### Euclidean Distance Kernels
 
# Squared distance between vectors x and y
function sqdist{T<:FloatingPoint}(x::Array{T}, y::Array{T})
    (n = length(x)) == length(y) || throw(ArgumentError("Dimensions do not conform."))
    c = zero(T)
    @inbounds @simd for i = 1:n
        v = x[i] - y[i]
        c += v*v
    end
    c
end

# Weighted squared distance function between vectors x and y
function sqdist{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    c = zero(T)
    @inbounds @simd for i = 1:n
        v = (x[i] - y[i]) * w[i]
        c += v*v
    end
    c
end

dsqdist_dx{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = scale!(2, x - y)
dsqdist_dy{T<:FloatingPoint}(x::Array{T}, y::Array{T}) = scale!(2, y - x)

function dsqdist_dx!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    @inbounds @simd for i = 1:n
        x[i] = 2(x[i] - y[i]) * w[i]^2
    end
    x
end

dsqdist_dy!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = dsqdist_dx!(y, x, w)

function dsqdist_dw!{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T})
    (n = length(x)) == length(y) == length(w) || throw(ArgumentError("Dimensions do not conform."))
    @inbounds @simd for i = 1:n
        w[i] = 2(x[i] - y[i])^2 * w[i]
    end
    w
end

dsqdist_dx{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = dsqdist_dx!(copy(x), y, w)
dsqdist_dy{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = dsqdist_dy!(x, copy(y), w)
dsqdist_dw{T<:FloatingPoint}(x::Array{T}, y::Array{T}, w::Array{T}) = dsqdist_dw!(x, y, copy(w))

