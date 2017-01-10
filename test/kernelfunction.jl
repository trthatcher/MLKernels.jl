#= Test Constructors =#

for k in kernel_functions
    info("Testing ", k)
    def_args, alt_args = get(kernel_functions_arguments, k, ((), ()))
    for T in FloatingPointTypes
        for i = 1:length(alt_args)
            K = (k)([typeof(θ) <: AbstractFloat ? convert(T,θ) : θ for θ in alt_args[1:i]]...)
            for j = 1:length(alt_args)
                @test getfield(K,j).value == (j <= i ? alt_args[j] : def_args[j])
            end
        end
    end
    kf = get(kernel_functions_kappa, k, x->error(""))
    for T in FloatingPointTypes
        def_args_T = [typeof(θ) <: AbstractFloat ? convert(T,θ) : θ for θ in def_args]
        K = (k)(def_args_T...)
        for z in (zero(T), one(T), convert(T,2))
            v = MOD.kappa(K, z)
            v_tmp = (kf)(z, def_args_T...)
            @test_approx_eq v v_tmp
        end
    end
end
