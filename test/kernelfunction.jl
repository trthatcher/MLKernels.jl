n = 200
m = 100
p = 25

info("Testing ", MOD.kernel)
for kernelobj in (additive_kernels..., composition_kernels...)
    for spX in (true,false), spY in (true,false)
        for T in FloatingPointTypes
            x = spX ? convert(SparseMatrixCSC{T},sprand(p,1,0.2)) : rand(T,p)
            y = spY ? convert(SparseMatrixCSC{T},sprand(p,1,0.2)) : rand(T,p)
            k = convert(Kernel{T}, (kernelobj)())
           
            if isa(k, PairwiseKernel)
                @test MOD.kernel(k, x[1], y[1]) == MOD.phi(k, x[1], y[1])
                @test MOD.kernel(k, x, y)       == sum(map((x,y) -> MOD.phi(k,x,y), x, y))
            else
                @test MOD.kernel(k, x[1], y[1]) == MOD.phi(k.phi, MOD.phi(k.kappa, x[1], y[1]))
                @test MOD.kernel(k, x, y)       == MOD.phi(k.phi, sum(map((x,y) -> MOD.phi(k.kappa,x,y), x, y)))
            end
        end
    end
end

info("Testing ", MOD.kernelmatrix)
for kernelobj in (additive_kernels..., composition_kernels...)
    for spX in (true,false), spY in (true,false)
        for T in FloatingPointTypes
            Set_X = spX ? [convert(SparseMatrixCSC{T},sprand(p,1,0.2)) for i = 1:n] :
                          [rand(T, p) for i = 1:n]
            Set_Y = spY ? [convert(SparseMatrixCSC{T},sprand(p,1,0.2)) for i = 1:m] :
                          [rand(T,p) for i = 1:m]

            X = transpose(hcat(Set_X...))
            Y = transpose(hcat(Set_Y...))
            k = convert(Kernel{T}, (kernelobj)())
            #println("κ:", k, ", X:", typeof(X), ", Y:", typeof(Y))
            
            P = [MOD.kernel(k,x,y) for x in Set_X, y in Set_X]
            @test_approx_eq MOD.kernelmatrix!(Val{:row}, Array(T,n,n), k, X)  P
            @test_approx_eq MOD.kernelmatrix!(Val{:col}, Array(T,n,n), k, X') P
            @test_approx_eq MOD.kernelmatrix(Val{:row}, k, X)  P
            @test_approx_eq MOD.kernelmatrix(Val{:col}, k, X') P
            @test_approx_eq MOD.kernelmatrix(k, X) P

            P = [MOD.kernel(k,x,y) for x in Set_X, y in Set_Y]
            @test_approx_eq MOD.kernelmatrix!(Val{:row}, Array(T,n,m), k, X,  Y)  P
            @test_approx_eq MOD.kernelmatrix!(Val{:col}, Array(T,n,m), k, X', Y') P
            @test_approx_eq MOD.kernelmatrix(Val{:row}, k, X,  Y)  P
            @test_approx_eq MOD.kernelmatrix(Val{:col}, k, X', Y') P
            @test_approx_eq MOD.kernelmatrix(k, X,  Y) P
        end
    end
end

#=
info("Testing ", centerkernelmatrix)
for T in (Float32, Float64)
    A = T[1 2 3;
          2 3 4;
          3 4 5]

    a = mean(A,1)

    @test_approx_eq centerkernelmatrix(A) ((A .- a) .- a') .+ mean(A)
end
=#
