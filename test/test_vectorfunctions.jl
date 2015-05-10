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
@test MLKernels.scprod([1,2.0], [1,3.0], [1,2.0]) == 13
println("Done")

x1 = [1.0; 2]
x2 = [2.0; 0]
X = [x1'; x2']
y1 = [1.0; 1]
y2 = [1.0; 1]
Y = [y1'; y2']
w = [2.0; 1]

print("- Testing scprodmatrix ... ")
Z = [MLKernels.scprod(x,y) for x in (x1,x2), y in (x1,x2)]; matrix_test_approx_eq(scprodmatrix(X), Z)
Z = [MLKernels.scprod(x,y) for x in (x1,x2), y in (y1,y2)]; matrix_test_approx_eq(scprodmatrix(X, Y), Z)
Z = [MLKernels.scprod(x,y,w) for x in (x1,x2), y in (x1,x2)]; matrix_test_approx_eq(scprodmatrix(X, w), Z)
Z = [MLKernels.scprod(x,y,w) for x in (x1,x2), y in (y1,y2)]; matrix_test_approx_eq(scprodmatrix(X, Y, w), Z)
println("Done")

print("- Testing sqdistmatrix ... ")
Z = [MLKernels.sqdist(x,y) for x in (x1,x2), y in (x1,x2)]; matrix_test_approx_eq(sqdistmatrix(X), Z)
Z = [MLKernels.sqdist(x,y) for x in (x1,x2), y in (y1,y2)]; matrix_test_approx_eq(sqdistmatrix(X, Y), Z)
Z = [MLKernels.sqdist(x,y,w) for x in (x1,x2), y in (x1,x2)]; matrix_test_approx_eq(sqdistmatrix(X, w), Z)
Z = [MLKernels.sqdist(x,y,w) for x in (x1,x2), y in (y1,y2)]; matrix_test_approx_eq(sqdistmatrix(X, Y, w), Z)
println("Done")

print("- Testing in-place scprod ... ")
Z = MLKernels.scprodmatrix(X,Y)
@test MLKernels.scprod(2,X,2,Y,1,false) == Z[2,1]
println("Done")

print("- Testing in-place sqdist ... ")
Z = MLKernels.sqdistmatrix(X,Y)
@test MLKernels.sqdist(2,X,2,Y,1,false) == Z[2,1]
println("Done")
