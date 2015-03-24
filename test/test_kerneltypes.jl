using Base.Test

importall MLKernels

println("-- Testing Kernel Scaling --")

for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]

    kernel = LinearKernel(one(T))
    skernel = ScaledKernel{T}(convert(T, 2),kernel)

    @test skernel.a == convert(T,2)
    @test skernel.κ == kernel

    show(skernel)
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

    @test (x -> true)(MLKernels.description_string(skernel))

end

println()
println("-- Testing Kernel Product --")

for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]

    kernel = LinearKernel(one(T))
    skernel = ScaledKernel(convert(T,2), kernel)
    kernelprod = KernelProduct{T}(convert(T, 2),kernel, LinearKernel(zero(T)))

    @test kernelprod.a == convert(T,2)
    @test kernelprod.κ₁ == kernel
    @test kernelprod.κ₂ == LinearKernel(zero(T))

    show(kernelprod)
    println()
    
    @test *(kernel, LinearKernel(1.0)).a == 1.0

    for S in (Float32, Float64, Int32, Int64)
        
        @test *(kernel, kernel).a == one(T)

        @test *(skernel, kernel).a == convert(T, 2)
        @test *(kernel, skernel).a == convert(T, 2)

        @test *(skernel, skernel).a == convert(T, 4)

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

    @test (x -> true)(MLKernels.description_string(kernelprod))

end

println()
println("-- Testing Kernel Sum --")

for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]

    kernel = LinearKernel(one(T))
    skernel = ScaledKernel(convert(T,2), kernel)
    kernelsum = KernelSum{T}(one(T),kernel, convert(T,2), LinearKernel(zero(T)))

    @test kernelsum.a₁ == one(T)
    @test kernelsum.κ₁ == kernel
    @test kernelsum.a₂ == convert(T,2)
    @test kernelsum.κ₂ == LinearKernel(zero(T))

    show(kernelsum)
    println()

    @test (+(kernel, kernel)).a₁ == one(T)
    @test (+(kernel, kernel)).a₂ == one(T)
        
    @test (+(skernel, kernel)).a₁ == convert(T, 2)
    @test (+(skernel, kernel)).a₂ == one(T)

    @test (+(kernel, skernel)).a₁ == one(T)
    @test (+(kernel, skernel)).a₂ == convert(T, 2)

    @test (+(skernel, skernel)).a₁ == convert(T, 2)
    @test (+(skernel, skernel)).a₂ == convert(T, 2)

    for S in (Float32, Float64, Int32, Int64)

        @test (*(convert(S,2), kernelsum)).a₁ == convert(T,2)
        @test (*(convert(S,2), kernelsum)).κ₁ == LinearKernel(one(promote_type(T,S)))
        @test (*(convert(S,2), kernelsum)).a₂ == convert(T,4)
        @test (*(convert(S,2), kernelsum)).κ₂ == LinearKernel(zero(promote_type(T,S)))

        @test (*(kernelsum, convert(S,2))).a₁ == convert(T,2)
        @test (*(kernelsum, convert(S,2))).κ₁ == LinearKernel(one(promote_type(T,S)))
        @test (*(kernelsum, convert(S,2))).a₂ == convert(T,4)
        @test (*(kernelsum, convert(S,2))).κ₂ == LinearKernel(zero(promote_type(T,S)))

    end

    @test kernel_function(kernelsum, x ,y) == convert(T, 4)
    @test isposdef_kernel(kernelsum) == true
    @test eltype(kernelsum) == T

    for S in (Float32, Float64)
        @test convert(KernelSum{S}, kernelsum).κ₁ == LinearKernel(one(S))
        @test convert(CompositeKernel{S}, kernelsum).κ₁ == LinearKernel(one(S))
        @test convert(Kernel{S}, kernelsum).κ₁ == LinearKernel(one(S))
    end

end

