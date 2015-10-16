#===================================================================================================
  Kernel Matrix Approximation
===================================================================================================#

const liblapack = Base.liblapack_name

import Base.blasfunc

import Base.LinAlg: BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, chksquare

typealias BlasChar Char

#Generic LAPACK error handlers
macro assertargsok() #Handle only negative info codes - use only if positive info code is useful!
    :(info[1]<0 && throw(ArgumentError("invalid argument #$(-info[1]) to LAPACK call")))
end
macro lapackerror() #Handle all nonzero info codes
    :(info[1]>0 ? throw(LAPACKException(info[1])) : @assertargsok )
end

macro assertnonsingular()
    :(info[1]>0 && throw(SingularException(info[1])))
end
macro assertposdef()
    :(info[1]>0 && throw(PosDefException(info[1])))
end

#Check that upper/lower (for special matrices) is correctly specified
macro chkuplo()
    :((uplo=='U' || uplo=='L') || throw(ArgumentError("""invalid uplo = $uplo
Valid choices are 'U' (upper) or 'L' (lower).""")))
end

for (syevd, elty) in
    ((:dsyevd_, :Float64),
     (:ssyevd_, :Float32))
    @eval begin
        #       SUBROUTINE dsyevd( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, IWORK, LIWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDA, LIWORK, LWORK, N
        # *     .. Array Arguments ..
        #       INTEGER            IWORK( * )
        #       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * )
        function syevd!(jobz::BlasChar, uplo::BlasChar, A::StridedMatrix{$elty})
            chkstride1(A)
            n = chksquare(A)
            W     = similar(A, $elty, n)
            work  = Array($elty, 1)
            lwork = convert(BlasInt, -1)
            iwork  = Array(BlasInt, 1)
            liwork = convert(BlasInt, -1)
            info  = Array(BlasInt, 1)
            for i in 1:2
                ccall(($(blasfunc(syevd)), liblapack), Void,
                     (Ptr{BlasChar}, Ptr{BlasChar}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                      Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                      &jobz, &uplo, &n, A, &max(1,stride(A,2)), W, work, &lwork, iwork, &liwork, info)
                @lapackerror
                if lwork < 0
                    lwork = convert(BlasInt, real(work[1]))
                    work = Array($elty, lwork)
                    liwork = iwork[1]
                    iwork = Array(BlasInt, liwork)
                end
            end
            jobz=='V' ? (W, A) : W
        end
    end
end


#===================================================================================================
  Nystrom Method
===================================================================================================#

# Nystrom method for Kernel Matrix approximation
function nystrom!{T<:AbstractFloat,U<:Integer}(K::Matrix{T}, κ::Kernel{T}, X::Matrix{T}, s::Vector{U}, is_trans::Bool, store_upper::Bool, symmetrize::Bool)
    c = length(s)
    n = size(X, 1)
    C = is_trans ? kernelmatrix(κ, X[:,s], X, true) : kernelmatrix(κ, X[s,:], X, false)
    D, V = syevd!('V', 'U', is_trans ? C[:,s] : C[s,:])
    tol = eps(T)*c
    @inbounds for i = 1:c
        D[i] = D[i] < tol ? zero(T) : 1/sqrt(D[i])
    end
    BLAS.syrk!(store_upper ? 'U' : 'L', 'N', one(T), BLAS.gemm('T', 'N', C, scale!(V, D)), zero(T), K)
    symmetrize ? (store_upper ?  syml!(K) : symu!(K)) : K
end

function nystrom{T<:AbstractFloat,U<:Integer}(κ::Kernel{T}, X::Matrix{T}, s::Array{U}, is_trans::Bool = false, store_upper::Bool = true, symmetrize::Bool = true)
    nystrom!(init_pairwise(X, is_trans), κ, X, s, is_trans, store_upper, symmetrize)
end

function nystrom{T<:AbstractFloat,U<:Integer}(κ::Kernel{T}, X::Matrix{T}, s::Array{U}; is_trans::Bool = false, store_upper::Bool = true, symmetrize::Bool = true)
    nystrom(κ, X, s, is_trans, store_upper, symmetrize)
end
