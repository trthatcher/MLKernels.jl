using Base.Test

importall MLKernels

print("- Testing matrix functions ... ")
for T in (Float32,Float64)
    S1 = T[1 2 3;
           4 5 6;
           7 8 9]
    SL = T[1 2 3;
           2 5 6;
           3 6 9]
    SU = T[1 4 7;
           4 5 8;
           7 8 9]
    dr = T[1*1+2*2+3*3, 77, 194]
    dc = T[1*1+4*4+7*7, 93, 126]
    row = T[11 12 13]
    col = T[11;12;13]
    RA = T[12 14 16;
           15 17 19;
           18 20 22]
    CA = T[12 13 14;
           16 17 18;
           20 21 22]
    RS = T[-10 -10 -10;
           -7 -7 -7;
           -4 -4 -4]
    CS = T[-10 -9 -8;
           -8 -7 -6;
           -6 -5 -4]
    diag = T[1, 2, 3]
    DA = T[1 2 3;
           8 10 12;
           21 24 27]
    AD = T[1 4 9;
           4 10 18;
           7 16 27]
    @test MLKernels.syml(S1) == SL
    @test MLKernels.symu(S1) == SU
    @test MLKernels.dot_rows(S1) == dr
    @test MLKernels.dot_columns(S1) == dc
    @test MLKernels.row_add!(copy(S1), row) == RA
    @test MLKernels.col_add!(copy(S1), col) == CA
    @test MLKernels.row_sub!(copy(S1), row) == RS
    @test MLKernels.col_sub!(copy(S1), col) == CS
    @test MLKernels.dgmm(diag, S1) == DA
    @test MLKernels.gdmm(S1, diag) == AD
    @test MLKernels.sqdist([1,2.0], [1 3.0]) == 1
    @test MLKernels.sqdist([1,2.0], [1,3.0], [1,2.0]) == 4
    @test MLKernels.scprod([1,2.0], [1 3.0]) == 7
    @test MLKernels.scprod([1,2.0], [1,3.0], [1,2.0]) == 13
    S = T[2 2 2;
          2 2 2;
          2 2 2]
    @test MLKernels.perturb(S, one(T)) == T[3 2 2; 2 3 2; 2 2 3]
    @test MLKernels.regularize(S, one(T), one(T)) == T[1 0 0; 0 1 0; 0 0 1]
    A = T[1 1 1;
          2 2 2]
    B = T[2 2 2;
          3 3 3]
    C1 = reshape(T[-1 -1 -1 0 0 0 -2 -2 -2 -1 -1 -1], (3,2,2))  # block_X == true
    C2 = reshape(T[-1 -1 -1 -2 -2 -2 -0 -0 -0 -1 -1 -1], (3,2,2))  # block_X == false
    @test MLKernels.sqdistmatrix_dx(one(T), A, B, 'N', true) == 2*C1
    @test MLKernels.sqdistmatrix_dx(one(T), A', B', 'T', true) == 2*C1
    @test MLKernels.sqdistmatrix_dx(one(T), A, B, 'N', false) == 2*C2
    @test MLKernels.sqdistmatrix_dx(one(T), A', B', 'T', false) == 2*C2
    @test MLKernels.sqdistmatrix_dx(convert(T,2), A, B, 'N', true) == 4*C1
    @test MLKernels.sqdistmatrix_dx(convert(T,2), A', B', 'T', true) == 4*C1
    @test MLKernels.sqdistmatrix_dx(convert(T,2), A, B, 'N', false) == 4*C2
    @test MLKernels.sqdistmatrix_dx(convert(T,2), A', B', 'T', false) == 4*C2

end
println("Done")
