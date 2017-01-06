for k in kernel_functions
    info("Testing ", k)
    for T in FloatingPointTypes
        def_args, alt_args = get(kernel_functions_arguments, k, ((), ()))
        for i = 1:length(alt_args)
            K = (k)([typeof(θ) <: AbstractFloat ? convert(T,θ) : θ for θ in alt_args[1:i]]...)
            for j = 1:length(alt_args)
                @test getfield(K,j).value == (j <= i ? alt_args[j] : def_args[j])
            end
        end
    end
end
