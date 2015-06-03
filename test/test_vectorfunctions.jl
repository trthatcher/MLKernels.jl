using Base.Test
importall MLKernels

function matrix_test_approx_eq(A::Array, B::Array)
    length(A) == length(B) || error("dimensions do not conform")
    for i = 1:length(A)
        @test_approx_eq A[i] B[i]
    end
end

print("- Testing sqdist ...")
@test MLKernels.sqdist([1,2.0], [1 3.0]) == 1
@test MLKernels.sqdist([1,2.0], [1,3.0], [1,2.0]) == 4
println("Done")

print("- Testing scprod ...")
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
