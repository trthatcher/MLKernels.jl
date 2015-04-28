using Base.Test

importall MLKernels

# Test Scaled Kernel
print("- Testing ScaledKernel constructors ... ")
for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]

    kernel = LinearKernel(one(T))
    skernel = ScaledKernel{T}(convert(T, 2),kernel)

    @test skernel.a == convert(T,2)
    @test skernel.k == kernel
    for S in (Float32, Float64, Int32, Int64)
        @test ScaledKernel(convert(S,2), kernel).a == convert(T, 2)
        @test *(convert(S,2), kernel).a == convert(T, 2)
        @test *(kernel, convert(S,2)).a == convert(T, 2)
        @test *(convert(S,2), skernel).a == convert(T, 4)
        @test *(skernel, convert(S,2)).a == convert(T, 4)
    end
end
println("Done")

print("- Testing ScaledKernel miscellaneous functions ... ")
for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]
    skernel = ScaledKernel{T}(convert(T, 2),LinearKernel(one(T)))
    @test kernel(skernel, x ,y) == convert(T, 4)
    @test isposdef(skernel) == true
    @test eltype(skernel) == T
    @test typeof(MLKernels.description_string(skernel)) <: String
end
println("Done")

print("- Testing ScaledKernel conversions ... ")
for T in (Float32, Float64)
    skernel = ScaledKernel{T}(convert(T, 2),LinearKernel(one(T)))
    for S in (Float32, Float64)
        @test convert(ScaledKernel{S}, skernel).k == LinearKernel(one(S))
        @test convert(SimpleKernel{S}, skernel).k == LinearKernel(one(S))
        @test convert(Kernel{S}, skernel).k == LinearKernel(one(S))
    end
end
println("Done")


# Test KernelProduct
print("- Testing KernelProduct constructors ... ")
for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]
    kernel = LinearKernel(one(T))
    skernel = ScaledKernel(convert(T,2), kernel)
    kernelprod = KernelProduct{T}(convert(T, 2),kernel, LinearKernel(zero(T)))

    @test kernelprod.a == convert(T,2)
    @test kernelprod.k1 == kernel
    @test kernelprod.k2 == LinearKernel(zero(T))
    
    @test *(kernel, LinearKernel(1.0)).a == 1.0

    for S in (Float32, Float64, Int32, Int64)
        @test *(kernel, kernel).a == one(T)

        @test *(skernel, kernel).a == convert(T, 2)
        @test *(kernel, skernel).a == convert(T, 2)

        @test *(skernel, skernel).a == convert(T, 4)

        @test *(kernelprod, convert(S,2)).a == convert(T, 4)
        @test *(convert(S,2), kernelprod).a == convert(T, 4)
    end
end
println("Done")

print("- Testing KernelProduct miscellaneous functions ...")
for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]
    kernelprod = KernelProduct{T}(convert(T, 2),LinearKernel(one(T)), LinearKernel(zero(T)))

    @test kernel(kernelprod, x ,y) == convert(T, 4)
    @test isposdef(kernelprod) == true
    @test eltype(kernelprod) == T
    @test typeof(MLKernels.description_string(kernelprod)) <: String
end
println("Done")

print("- Testing KernelProduct conversions ... ")
for T in (Float32, Float64)
    kernelprod = KernelProduct{T}(convert(T, 2), LinearKernel(one(T)), LinearKernel(zero(T)))
    for S in (Float32, Float64)
        @test convert(KernelProduct{S}, kernelprod).k1 == LinearKernel(one(S))
        @test convert(CompositeKernel{S}, kernelprod).k1 == LinearKernel(one(S))
        @test convert(Kernel{S}, kernelprod).k1 == LinearKernel(one(S))
    end
end
println("Done")

# Test KernelSum
print("- Testing KernelSum constructors ... ")
for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]
    kernel = LinearKernel(one(T))
    skernel = ScaledKernel(convert(T,2), kernel)
    kernelsum = KernelSum{T}(one(T),kernel, convert(T,2), LinearKernel(zero(T)))

    @test kernelsum.a1 == one(T)
    @test kernelsum.k1 == kernel
    @test kernelsum.a2 == convert(T,2)
    @test kernelsum.k2 == LinearKernel(zero(T))

    @test (+(kernel, kernel)).a1 == one(T)
    @test (+(kernel, kernel)).a2 == one(T)
        
    @test (+(skernel, kernel)).a1 == convert(T, 2)
    @test (+(skernel, kernel)).a2 == one(T)

    @test (+(kernel, skernel)).a1 == one(T)
    @test (+(kernel, skernel)).a2 == convert(T, 2)

    @test (+(skernel, skernel)).a1 == convert(T, 2)
    @test (+(skernel, skernel)).a2 == convert(T, 2)

    for S in (Float32, Float64, Int32, Int64)

        @test (*(convert(S,2), kernelsum)).a1 == convert(T,2)
        @test (*(convert(S,2), kernelsum)).k1 == LinearKernel(one(promote_type(T,S)))
        @test (*(convert(S,2), kernelsum)).a2 == convert(T,4)
        @test (*(convert(S,2), kernelsum)).k2 == LinearKernel(zero(promote_type(T,S)))

        @test (*(kernelsum, convert(S,2))).a1 == convert(T,2)
        @test (*(kernelsum, convert(S,2))).k1 == LinearKernel(one(promote_type(T,S)))
        @test (*(kernelsum, convert(S,2))).a2 == convert(T,4)
        @test (*(kernelsum, convert(S,2))).k2 == LinearKernel(zero(promote_type(T,S)))

    end
end
println("Done")

print("- Testing KernelSum miscellaneous functions ...")
for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]
    kernelsum = KernelSum{T}(one(T),LinearKernel(one(T)), convert(T,2), LinearKernel(zero(T)))
    @test kernel(kernelsum, x ,y) == convert(T, 4)
    @test isposdef(kernelsum) == true
    @test eltype(kernelsum) == T
end
println("Done")

print("- Testing KernelSum conversions ... ")
for T in (Float32, Float64)
    kernelsum = KernelSum{T}(one(T),LinearKernel(one(T)), convert(T,2), LinearKernel(zero(T)))
    for S in (Float32, Float64)
        @test convert(KernelSum{S}, kernelsum).k1 == LinearKernel(one(S))
        @test convert(CompositeKernel{S}, kernelsum).k1 == LinearKernel(one(S))
        @test convert(Kernel{S}, kernelsum).k1 == LinearKernel(one(S))
    end
end
println("Done")

# Show output
println("- Testing composite kernel output: ")

    print("    - Testing ")
    show(STDOUT, 2*LinearKernel())
    println(" ... Done")
 
    print("    - Testing ")
    show(STDOUT, 2*LinearKernel()*LinearKernel())
    println(" ... Done")

    print("    - Testing ")
    show(STDOUT, 2*LinearKernel() + 1*LinearKernel())
    println(" ... Done")

