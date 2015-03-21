using Base.Test

println("Test Kernel Scaling")

for T in (Float32, Float64)
    x, y = [one(T)], [one(T)]

    kernel = LinearKernel(one(T))
    s2kernel = ScaledKernel{T}(convert(T, 2),kernel)
    @test show(s2kernel) == Nothing()
    s4kernel = ScaledKernel{T}(convert(T, 4),kernel)

    for S in (Float32, Float64, Int32, Int64)
        @test ScaledKernel(convert(T,2), kernel).a == s2kernel.a
        @test *(convert(T,2), kernel).a == s2kernel.a
        @test *(kernel, convert(T,2)).a == s2kernel.a
        @test *(convert(T,2), s2kernel).a == s4kernel.a
        @test *(s2kernel, convert(T,2)).a == s4kernel.a
    end

    @test kernel_function(s2kernel, x ,y) == convert(T, 4)
    @test isposdef_kernel(s2kernel) == true

end


