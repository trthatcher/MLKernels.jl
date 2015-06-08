#===================================================================================================
  Kernel Matrix Approximation
===================================================================================================#

const liblapack = Base.liblapack_name

import Base.blasfunc

import Base.LinAlg: BlasFloat, BlasChar, BlasInt, blas_int, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, chksquare

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
            lwork = blas_int(-1)
            iwork  = Array(BlasInt, 1)
            liwork = blas_int(-1)
            info  = Array(BlasInt, 1)
            for i in 1:2
                ccall(($(blasfunc(syevd)), liblapack), Void,
                     (Ptr{BlasChar}, Ptr{BlasChar}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                      Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                      &jobz, &uplo, &n, A, &max(1,stride(A,2)), W, work, &lwork, iwork, &liwork, info)
                @lapackerror
                if lwork < 0
                    lwork = blas_int(real(work[1]))
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

# Moore-Penrose pseudo-inverse for positive semidefinite matrices
function nystrom_decomp!{T<:FloatingPoint}(S::Matrix{T}, uplo::Char = 'U', tol::T = eps(T)*maximum(size(S)))
    tol > 0 || error("tol = $tol must be a positive number.")
    (n = size(S, 1)) == size(S, 2) || throw(ArgumentError("S must be a square matrix"))
    D, V = syevd!('V', uplo, S)
	@inbounds for i = 1:n
		if D[i] < tol
            D[i] = zero(T)
        else
            D[i] = 1/sqrt(D[i])
        end
    end
    scale!(D, V)
end


# Nystrom method for Kernel Matrix approximation
function nystrom{T<:FloatingPoint,S<:Integer}(κ::Kernel{T}, X::Matrix{T}, s::Array{S})
    c = length(s)
    n = size(X, 1)
    C = kernelmatrix(κ, X[s,:], X)
    DV = nystrom_decomp!(C[:,s])
    DVC = BLAS.gemm('N', 'N', DV, C)
    K = BLAS.syrk('U', 'T', one(T), DVC)
    syml!(K)
end
