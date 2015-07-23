print("- Testing matrix functions ... ")
for T in (Float32,Float64,BigFloat)
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
    w = T[1;2;3]
    wdr = vec(sum((S1 .* S1) .* w',2))
    wdc = vec(sum((S1 .* S1) .* w,1))
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
    @test MLKernels.syml(S1) == SL
    @test MLKernels.symu(S1) == SU
    @test MLKernels.dot_rows(S1) == dr
    @test MLKernels.dot_columns(S1) == dc
    @test MLKernels.dot_rows(S1,w) == wdr
    @test MLKernels.dot_columns(S1,w) == wdc
    S = T[2 2 2;
          2 2 2;
          2 2 2]
    @test MLKernels.matrix_prod!(T[3; 2], T[3; 2]) == T[9; 4]
    @test MLKernels.matrix_sum!(T[3; 2], T[3; 2]) == T[6; 4]
    @test MLKernels.translate!(T[3; 2], one(T)) == T[4; 3]
    @test MLKernels.translate!(one(T), T[3; 2]) == T[4; 3]

end
println("Done")

#=
print("- Testing sqdist ... ")
@test MLKernels.sqdist(2.0, 3.0) == 1
@test MLKernels.sqdist(2.0, 3.0, 2.0) == 4
@test MLKernels.sqdist([1,2.0], [1 3.0]) == 1
@test MLKernels.sqdist([1,2.0], [1,3.0], [1,2.0]) == 4
println("Done")

print("- Testing scprod ... ")
@test MLKernels.scprod(2.0, 3.0) == 6
@test MLKernels.scprod(2.0, 3.0, 2.0) == 24
@test MLKernels.scprod([1,2.0], [1 3.0]) == 7
@test MLKernels.scprod([1,2.0], [1,3.0], [1,2.0]) == 25
println("Done")

x1 = [1.0; 2]
x2 = [2.0; 0]
x3 = [3.0; 2]
X = [x1'; x2'; x3']
y1 = [1.0; 1]
y2 = [1.0; 1]
Y = [y1'; y2']
w = [2.0; 1]

Sx = (x1,x2,x3)
Sy = (y1,y2)

print("- Testing square scprodmatrix ... ")
Z = [MLKernels.scprod(x,y) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.scprodmatrix(X), Z)
Z = [MLKernels.scprod(x,y) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.scprodmatrix(X', true), Z)
println("Done")

print("- Testing rectangular scprodmatrix ... ")
Z = [MLKernels.scprod(x,y) for x in Sx, y in Sy]; matrix_test_approx_eq(MLKernels.scprodmatrix(X, Y), Z)
Z = [MLKernels.scprod(x,y) for x in Sx, y in Sy]; matrix_test_approx_eq(MLKernels.scprodmatrix(X', Y', true), Z)
println("Done")

print("- Testing weighted square scprodmatrix ... ")
Z = [MLKernels.scprod(x,y,w) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.scprodmatrix(X, w), Z)
Z = [MLKernels.scprod(x,y,w) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.scprodmatrix(X', w, true), Z)
println("Done")

print("- Testing weighted rectangular scprodmatrix ... ")
Z = [MLKernels.scprod(x,y,w) for x in Sx, y in Sy]; matrix_test_approx_eq(MLKernels.scprodmatrix(X, Y, w), Z)
Z = [MLKernels.scprod(x,y,w) for x in Sx, y in Sy]; matrix_test_approx_eq(MLKernels.scprodmatrix(X', Y', w, true), Z)
println("Done")

print("- Testing square sqdistmatrix ... ")
Z = [MLKernels.sqdist(x,y) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X), Z)
Z = [MLKernels.sqdist(x,y) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X, false, false), Z) # is_upper=false
Z = [MLKernels.sqdist(x,y) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X', true), Z)
Z = [MLKernels.sqdist(x,y) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X', true, false), Z) # is_upper=false
println("Done")

print("- Testing rectangular sqdistmatrix ... ")
Z = [MLKernels.sqdist(x,y) for x in Sx, y in Sy]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X, Y), Z)
Z = [MLKernels.sqdist(x,y) for x in Sx, y in Sy]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X', Y', true), Z)
println("Done")

print("- Testing weighted square sqdistmatrix ... ")
Z = [MLKernels.sqdist(x,y,w) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X, w), Z)
Z = [MLKernels.sqdist(x,y,w) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X, w, false, false), Z) # is_upper=false
Z = [MLKernels.sqdist(x,y,w) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X', w, true), Z)
Z = [MLKernels.sqdist(x,y,w) for x in Sx, y in Sx]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X', w, true, false), Z) # is_upper=false
println("Done")

print("- Testing weighted rectangular sqdistmatrix ... ")
Z = [MLKernels.sqdist(x,y,w) for x in Sx, y in Sy]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X, Y, w), Z)
Z = [MLKernels.sqdist(x,y,w) for x in Sx, y in Sy]; matrix_test_approx_eq(MLKernels.sqdistmatrix(X', Y', w, true), Z)
println("Done")

print("- Testing in-place scprod ... ")
Z = MLKernels.scprodmatrix(X,Y)
@test MLKernels.scprod(2,X,2,Y,1,false) == Z[2,1]
Z = MLKernels.scprodmatrix(X,Y,w)
@test MLKernels.scprod(2,X,2,Y,1,w,false) == Z[2,1]

Z = MLKernels.scprodmatrix(X',Y',true)
@test MLKernels.scprod(2,X',2,Y',1,true) == Z[2,1]
Z = MLKernels.scprodmatrix(X',Y',w,true)
@test MLKernels.scprod(2,X',2,Y',1,w,true) == Z[2,1]
println("Done")

print("- Testing in-place sqdist ... ")
Z = MLKernels.sqdistmatrix(X,Y)
@test MLKernels.sqdist(2,X,2,Y,1,false) == Z[2,1]
Z = MLKernels.sqdistmatrix(X,Y,w)
@test MLKernels.sqdist(2,X,2,Y,1,w,false) == Z[2,1]

Z = MLKernels.sqdistmatrix(X',Y',true)
@test MLKernels.sqdist(2,X',2,Y',1,true) == Z[2,1]
Z = MLKernels.sqdistmatrix(X',Y',w,true)
@test MLKernels.sqdist(2,X',2,Y',1,w,true) == Z[2,1]
println("Done")
=#
