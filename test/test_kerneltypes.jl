using Base.Test

println("Test Kernel Scaling")

for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]

    kernel = LinearKernel(one(T))
    s2kernel = ScaledKernel{T}(convert(T, 2),kernel)
    @test show(s2kernel) == Nothing()
    s4kernel = ScaledKernel{T}(convert(T, 2),kernel)

    for S in (Float32, Float64, Int32, Int64)
        @test ScaledKernel(convert(T,2), kernel) == s2kernel
        @test *(convert(T,2), kernel) == s2kernel
        @test *(kernel, convert(T,2)) == s2kernel
        @test *(convert(T,2), s2kernel) == s4kernel
        @test *(s2kernel, convert(T,2)) == s4kernel
    end

    @test kernel_function(s2kernel, x ,y) == convert(T, 2)
    @test isposdef_kernel(s2kernel) == true

end


