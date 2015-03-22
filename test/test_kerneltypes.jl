using Base.Test

importall MLKernels

println("Test Kernel Scaling")

for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]

    kernel = LinearKernel(one(T))
    skernel = ScaledKernel{T}(convert(T, 2),kernel)

    @test skernel.a == convert(T,2)
    @test skernel.κ == kernel

    @test show(skernel) == Nothing()
    println()

    for S in (Float32, Float64, Int32, Int64)
        @test ScaledKernel(convert(S,2), kernel).a == convert(T, 2)
        @test *(convert(S,2), kernel).a == convert(T, 2)
        @test *(kernel, convert(S,2)).a == convert(T, 2)
        @test *(convert(S,2), skernel).a == convert(T, 4)
        @test *(skernel, convert(S,2)).a == convert(T, 4)
    end

    @test kernel_function(skernel, x ,y) == convert(T, 4)
    @test isposdef_kernel(skernel) == true
    @test eltype(skernel) == T

    for S in (Float32, Float64)
        @test convert(ScaledKernel{S}, skernel).κ == LinearKernel(one(S))
        @test convert(SimpleKernel{S}, skernel).κ == LinearKernel(one(S))
        @test convert(Kernel{S}, skernel).κ == LinearKernel(one(S))
    end

    @test show(skernel) == Nothing()
    @test (x -> true)(MLKernels.description_string(skernel))

end

for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]

    kernel = LinearKernel(one(T))
    skernel = ScaledKernel(convert(T,2), kernel)
    kernelprod = KernelProduct{T}(convert(T, 2),kernel, LinearKernel(zero(T)))

    @test kernelprod.a == convert(T,2)
    @test kernelprod.κ₁ == kernel
    @test kernelprod.κ₂ == LinearKernel(zero(T))

    @test show(kernelprod) == Nothing()
    println()
    
    @test *(kernel, LinearKernel(1.0)).a == 1.0

    for S in (Float32, Float64, Int32, Int64)
        
        @test *(convert(S,2), kernel, kernel).a == convert(T, 2)
        @test *(kernel, kernel, convert(S,2)).a == convert(T, 2)
        @test *(kernel, convert(S,2), kernel).a == convert(T, 2)

        @test *(convert(S,2), skernel, kernel).a == convert(T, 4)
        @test *(kernel, skernel, convert(S,2)).a == convert(T, 4)
        @test *(kernel, convert(S,2), skernel).a == convert(T, 4)

        @test *(convert(S,2), skernel, skernel).a == convert(T, 8)
        @test *(skernel, skernel, convert(S,2)).a == convert(T, 8)
        @test *(skernel, convert(S,2), skernel).a == convert(T, 8)

        @test *(kernelprod, convert(S,2)).a == convert(T, 4)
        @test *(convert(S,2), kernelprod).a == convert(T, 4)

    end

    @test kernel_function(kernelprod, x ,y) == convert(T, 4)
    @test isposdef_kernel(kernelprod) == true
    @test eltype(kernelprod) == T

    for S in (Float32, Float64)
        @test convert(KernelProduct{S}, kernelprod).κ₁ == LinearKernel(one(S))
        @test convert(CompositeKernel{S}, kernelprod).κ₁ == LinearKernel(one(S))
        @test convert(Kernel{S}, kernelprod).κ₁ == LinearKernel(one(S))
    end

    @test show(kernelprod) == Nothing()
    @test (x -> true)(MLKernels.description_string(kernelprod))

end



