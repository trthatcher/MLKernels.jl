info("Testing ", MOD.pairwise)
for kernelobj in additive_kernels
    for T in (Float64,)
        x1 = T[1; 2]
        x2 = T[2; 0]
        x3 = T[3; 2]
        X = [x1'; x2'; x3']

        y1 = T[1; 1]
        y2 = T[1; 1]
        Y = [y1'; y2']

        Set_x = (x1,x2,x3)
        Set_y = (y1,y2)
        k = (kernelobj)()
       
        @test MOD.pairwise(k, x1[1], y1[1])  == MOD.phi(k, x1[1], y1[1])
        @test MOD.pairwise(k, x1, y1) == MOD.phi(k, x1[1], y1[1]) + MOD.phi(k, x1[2], y1[2])
        
        P = [MOD.pairwise(k,x,y) for x in Set_x, y in Set_x]
        @test_approx_eq MOD.pairwise(Val{:row}, k, X)  P
        @test_approx_eq MOD.pairwise(Val{:col}, k, X') P

        P = [MOD.pairwise(k,x,y) for x in Set_x, y in Set_y]
        @test_approx_eq MOD.pairwise(Val{:row}, k, X, Y)   P
        @test_approx_eq MOD.pairwise(Val{:col}, k, X', Y') P
    end
end

#=
immutable TestKernel{T<:AbstractFloat} <: BaseKernel{T}
    a::T
end

MOD.phi{T<:AbstractFloat}(κ::TestKernel, x::AbstractVector{T}, y::AbstractVector{T}) = sum(x) + sum(y) + κ.a

function MOD.description_string{T<:AbstractFloat}(κ::TestKernel{T}, eltype::Bool = true)
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

    P = [MOD.pairwise(k,x,y) for x in Set_x, y in Set_x]

    @test_approx_eq MOD.syml!(MOD.pairwise(k, X,false,true))  P
    @test_approx_eq MOD.symu!(MOD.pairwise(k, X,false,false)) P

    @test_approx_eq MOD.syml!(MOD.pairwise(k, X',true,true))  P
    @test_approx_eq MOD.symu!(MOD.pairwise(k, X',true,false)) P

    P = [MOD.pairwise(k,x,y) for x in Set_x, y in Set_y]

    @test_approx_eq MOD.pairwise(k, X,  Y,  false)  P
    @test_approx_eq MOD.pairwise(k, X', Y', true)   P

end
=#
