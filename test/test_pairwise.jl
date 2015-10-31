info("Testing ", MLKernels.pairwise)
for kernelobj in additive_kernels
    for T in FloatingPointTypes
        x1 = T[1; 2]
        x2 = T[2; 0]
        x3 = T[3; 2]
        X = [x1'; x2'; x3']

        y1 = T[1; 1]
        y2 = T[1; 1]
        Y = [y1'; y2']

        w = T[2; 1]

        Set_x = (x1,x2,x3)
        Set_y = (y1,y2)

        k = convert(Kernel{T}, (kernelobj)())
       
        @test MLKernels.pairwise(k, x1[1], y1[1])             == MLKernels.phi(k, x1[1], y1[1])
        @test MLKernels.pairwise(ARD(k,w[1:1]), x1[1], y1[1]) == MLKernels.phi(k, x1[1], y1[1]) * w[1]^2

        @test MLKernels.pairwise(k, x1, y1)        == MLKernels.phi(k, x1[1], y1[1]) + 
                                                      MLKernels.phi(k, x1[2], y1[2])
        @test MLKernels.pairwise(ARD(k,w), x1, y1) == MLKernels.phi(k, x1[1], y1[1]) * w[1]^2 + 
                                                      MLKernels.phi(k, x1[2], y1[2]) * w[2]^2
        
        P = [MLKernels.pairwise(k,x,y) for x in Set_x, y in Set_x]
        #K = Array(T,length(Set_x),length(Set_x))

        @test_approx_eq MLKernels.syml!(MLKernels.pairwise(k, X,false,true))  P
        @test_approx_eq MLKernels.symu!(MLKernels.pairwise(k, X,false,false)) P

        @test_approx_eq MLKernels.syml!(MLKernels.pairwise(k, X',true,true))  P
        @test_approx_eq MLKernels.symu!(MLKernels.pairwise(k, X',true,false)) P

        P = [MLKernels.pairwise(k,x,y,w) for x in Set_x, y in Set_x]

        @test_approx_eq MLKernels.syml!(MLKernels.pairwise(ARD(k,w), X,false,true))  P
        @test_approx_eq MLKernels.symu!(MLKernels.pairwise(ARD(k,w), X,false,false)) P

        @test_approx_eq MLKernels.syml!(MLKernels.pairwise(ARD(k,w), X',true,true))  P
        @test_approx_eq MLKernels.symu!(MLKernels.pairwise(ARD(k,w), X',true,false)) P

        P = [MLKernels.pairwise(k,x,y) for x in Set_x, y in Set_y]

        @test_approx_eq MLKernels.pairwise(k, X, Y, false)  P
        @test_approx_eq MLKernels.pairwise(k, X', Y', true) P

        P = [MLKernels.pairwise(k,x,y,w) for x in Set_x, y in Set_y]

        @test_approx_eq MLKernels.pairwise(ARD(k,w), X, Y, false)  P
        @test_approx_eq MLKernels.pairwise(ARD(k,w), X', Y', true) P

    end
end


immutable TestKernel{T<:AbstractFloat} <: BaseKernel{T}
    a::T
end

MLKernels.phi{T<:AbstractFloat}(κ::TestKernel, x::AbstractVector{T}, y::AbstractVector{T}) = sum(x) + sum(y) + κ.a

function MLKernels.description_string{T<:AbstractFloat}(κ::TestKernel{T}, eltype::Bool = true)
    "Test" * (eltype ? "{$(T)}" : "") * "(a=$(κ.a))"
end

p = 10
for T in (Float32, Float64)
    x1 = rand(T,p)
    x2 = rand(T,p)
    x3 = rand(T,p)
    X = [x1'; x2'; x3']

    y1 = rand(T,p)
    y2 = rand(T,p)
    Y = [y1'; y2']

    Set_x = (x1, x2, x3)
    Set_y = (y1, y2)

    k = TestKernel(100*one(T))

    P = [MLKernels.pairwise(k,x,y) for x in Set_x, y in Set_x]

    @test_approx_eq MLKernels.syml!(MLKernels.pairwise(k, X,false,true))  P
    @test_approx_eq MLKernels.symu!(MLKernels.pairwise(k, X,false,false)) P

    @test_approx_eq MLKernels.syml!(MLKernels.pairwise(k, X',true,true))  P
    @test_approx_eq MLKernels.symu!(MLKernels.pairwise(k, X',true,false)) P

    P = [MLKernels.pairwise(k,x,y) for x in Set_x, y in Set_y]

    @test_approx_eq MLKernels.pairwise(k, X,  Y,  false)  P
    @test_approx_eq MLKernels.pairwise(k, X', Y', true)   P

end
