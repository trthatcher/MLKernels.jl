using Base.Test

importall MLKernels

const FloatingPointTypes = (Float32, Float64, BigFloat)

macro test_approx_eq_type(value, reference, typ)
    x = gensym()
    quote
        $x = $value
        @test_approx_eq $x $reference
        @test isa($x, $typ)
    end
end

# Check each field for equality with args (assumed same order)
function check_fields(kernelobject::BaseKernel, field_values)
    fields = names(kernelobject)
    for i = 1:length(fields)
        @test getfield(kernelobject, fields[i]) === field_values[i]
    end
end

# Iterate through constructor cases - test only Float64 to ensure fields are populating as expected
function test_constructor_case{T<:BaseKernel}(kernelobject::Type{T}, default_args, test_args, n = length(names(kernelobject)))
    check_fields((kernelobject)(), Float64[default_args...])
    for i = 1:n
        case_args = Float64[test_args[1:i]..., default_args[(i+1):end]...]
        κ = (kernelobject)(case_args[1:i]...)
        check_fields(κ, case_args)
    end
end

function test_constructor_case{T<:CompositeKernel}(kernelobject::Type{T}, default_args, test_args, n = length(names(kernelobject)))
    if n > 1
        default_kernel = default_args[1]
        default_params = default_args[2:n]
        check_fields((kernelobject)(), [convert(Kernel{Float64}, default_kernel),default_params...])
    end
    #check_fields((kernelobject)(), Float64[default_args...])
    #for i = 1:n
    #    case_args = Float64[test_args[1:i]..., default_args[(i+1):end]...]
    #    κ = (kernelobject)(case_args[1:i]...)
    #    check_fields(κ, case_args)
    #end
end




# Check each field for equality with args (assumed same order)
function check_fields(kernelobject::CompositeKernel, field_values)
    fields = names(kernelobject)
    @test kernelobject.k == field_values[1]
    if length(fields) > 1
        for i = 2:length(fields)
            @test getfield(kernelobject, fields[i]) === field_values[i]
        end
    end
end


# Compare the values of two kernels of the same type
#function check_fields{T<:StandardKernel}(kernel1::T, kernel2::T)
#    fields = names(kernel1)
#    for i = 1:length(fields)
#        @test getfield(kernel1, fields[i]) === getfield(kernel2, fields[i])
#    end
#end

function test_fields_equal(kernel1, kernel2)
    fields = names(kernel1)
    @test fields == names(kernel2)
    for i = 1:length(fields)
        f1 = getfield(kernel1, fields[i])
        f2 = getfield(kernel2, fields[i])
        if isa(f1, Kernel) && isa(f2, Kernel)
            test_fields_equal(f1, f2)
        elseif isa(f1, Array) && isa(f2, Array) && (eltype(f1) <: Kernel && eltype(f2) <: Kernel)
            @test length(f1) == length(f2)
            for i = 1:length(f1)
                test_fields_equal(f1[i], f2[i])
            end
        else
            @test f1 == f2
        end
    end
end


# Test constructor for argument error
function test_error_case(kernelobject, error_case)
    for T in FloatingPointTypes
        test_case = T[error_case...]
        @test_throws ArgumentError (kernelobject)(test_case...)
    end
end

function test_conversion(kernel)
    kerneltypes = [typeof(kernel), MLKernels.supertypes(typeof(kernel))...]
    fields = names(kernel)
    for S in FloatingPointTypes
        for convtype in kerneltypes
            converted = convert(convtype.name.primary{S}, kernel)
            @test eltype(converted) == S
            test_fields_equal(kernel, converted)
        end
    end
end

function test_kernel_function{T<:FloatingPoint}(K::Kernel, x::T, y::T, value::T)
    @test_approx_eq_type kernel(K, x, y)                       value T # scalar
    test_kernel_function(K, [x], [y], value)
end
function test_kernel_function{T<:FloatingPoint}(K::Kernel, x::Vector{T}, y::Vector{T}, value::T)
    @test_approx_eq_type kernel(K, x, y)                       value T # vector
    @test_approx_eq_type kernelmatrix(K, [x y]')[1,2]          value T # matrix X
    @test_approx_eq_type kernelmatrix(K, [x y], 'T')[1,2]      value T # matrix X'
    @test_approx_eq_type kernelmatrix(K, [x y]', 'N', 'L')[1,2] value T # matrix X'
    @test_approx_eq_type kernelmatrix(K, [x y], 'T', 'L')[1,2] value T # matrix X'
    @test_approx_eq_type kernelmatrix(K, x', y')[1,1]          value T # matrix X,Y
    @test_approx_eq_type kernelmatrix(K, x'', y'', 'T')[1,1]   value T # matrix X',Y'
end


# Test Base Kernels

info("Testing BaseKernel show()")
for kernelobject in (
        SquaredDistanceKernel,
        SineSquaredKernel,
        ChiSquaredKernel,
        ScalarProductKernel,
        MercerSigmoidKernel
    )
    show(STDOUT, (kernelobject)())
    println("")
end

info("Testing BaseKernel constructors")
for (kernelobject, default_args, test_args) in (
        (SquaredDistanceKernel, [1.0], [0.5]),
        (SineSquaredKernel, [1.0], [0.5]),
        (ChiSquaredKernel, [1.0], [0.5]),
        #(ScalarProductKernel,Float64[],Float64[])
        (MercerSigmoidKernel, [0.0, 1.0], [1.0, 2.0])
    )
    test_constructor_case(kernelobject, default_args, test_args)
end

info("Testing CompositeKernel constructors")
for (kernelobject, default_args, test_args) in (
        (ExponentialKernel, [SquaredDistanceKernel(), 1.0, 1.0], [ChiSquaredKernel(), 2.0,0.5]),
    )
    test_constructor_case(kernelobject, default_args, test_args)
end



#=
println("- Testing StandardKernel and ARD type conversions:")
for (kernelobject, default_args, test_args) in (
        (ExponentialKernel, [1, 1], [2, 0.5]),
        (RationalQuadraticKernel, [1, 1, 1], [2, 2, 0.5]),
        (PowerKernel, [1], [0.5]),
        (LogKernel, [1,1], [2,0.5]),
        (PolynomialKernel, [1,1,2], [2,2,3]),
        (MaternKernel, [1,1], [2,2]),
        (SigmoidKernel, [1,1], [2,2]),
    )
    print("    - Testing ", kernelobject, " ... ")
    for T in FloatingPointTypes
        args = T[test_args...]
        k = (kernelobject)(args...)
        test_conversion(k)
        test_conversion(ARD(k, T[1,2,3]))
    end
    println("Done")
end
println("- Testing CompositeKernel type conversions:")
for T in FloatingPointTypes
    k1 = ExponentialKernel(one(T))
    k2 = PolynomialKernel(one(T))
    a = convert(T, 3)
    test_conversion(a*k1*k2)
    test_conversion(a+k1+k2)
end

println("- Testing StandardKernel aliases:")
for (kernelobject, default_args, test_args) in (
        (GaussianKernel, [1, 1], [2]),
        (RadialBasisKernel, [1, 1], [2]),
        (LaplacianKernel, [1, 0.5], [2]),
        (LinearKernel, [1, 1, 1], [2, 2]),
    )
    print("    - Testing ", kernelobject, " ... ")
    test_constructor_case(kernelobject, default_args, test_args, length(test_args))
    println("Done")
end
for T in FloatingPointTypes
    k = PolynomialKernel(convert(T,2), convert(T,2), 2)
    check_fields(k, T[2, 2, 2])
end

println("- Testing ARD constructors:")
for (kernelobject, test_args) in (
        (ExponentialKernel, [2, 0.5]),
        (RationalQuadraticKernel, [2, 2, 0.5]),
        (PowerKernel, [0.5]),
        (LogKernel, [2,0.5]),
        (MaternKernel, [2,2]),
        (PolynomialKernel, [2,2,3]),
        (SigmoidKernel, [2,2]),
    )
    print("    - Testing ARD ", kernelobject, " ... ")
    for T in FloatingPointTypes
        w = [convert(T,2)]
        case_args = T[test_args...]
        K = ARD((kernelobject)(case_args...),w)
        @test K.w == w
        check_fields(K.k, case_args)
        d = 3
        K = ARD((kernelobject)(case_args...),d)
        @test K.w == ones(T,d)
        check_fields(K.k, case_args)
    end
    println("Done")
end


println("- Testing StandardKernel error cases:")
for (kernelobject, error_cases) in (
        (ExponentialKernel, ([0], [0, 1], [1, 0], [1, 2])),
        (RationalQuadraticKernel, ([0], [1, 0], [1, 1, 0], [1,1,1.01])),
        (PowerKernel, ([0],[1.0001])),
        (LogKernel, ([0],[1,0], [1,1.0001])),
        (MaternKernel, ([0], [1,0])),
        (PolynomialKernel, ([0,1,2], [1,-0.0001,3], [1,1,0])),
        (SigmoidKernel, ([0,1], [1,-0.00001])),
    )
    print("    - Testing ", kernelobject, " error cases ... ")
    for error_case in error_cases
        print(" ", error_case)
        test_error_case(kernelobject, error_case)
    end
    println(" ... Done")
end

println("- Testing ismercer() property:")
for (kernelobject, posdef) in (
        (ExponentialKernel, true),
        (RationalQuadraticKernel, true),
        (PowerKernel, false),
        (LogKernel, false),
        (MaternKernel, true),
        (PolynomialKernel, true),
        (SigmoidKernel, false),
    )
    print("    - Testing ", kernelobject, "... ")
    @test ismercer((kernelobject)()) == posdef
    println("Done")
end

println("- Testing iscondposdef() property:")
for (kernelobject, posdef) in (
        (ExponentialKernel, true),
        (RationalQuadraticKernel, true),
        (PowerKernel, true),
        (LogKernel, true),
        (MaternKernel, true),
        (PolynomialKernel, true),
        (SigmoidKernel, false),
    )
    print("    - Testing ", kernelobject, "... ")
    @test iscondposdef((kernelobject)()) == posdef
    println("Done")
end

println("- Testing ScalarProductKernel kernel() function:")
for (kernelobject, test_args, test_function) in (
        (PolynomialKernel, [1,1,2], (z,a,c,d) -> (a * z + c )^d),
        (SigmoidKernel, [1,1], (z,a,c) -> tanh(a*z + c))
    )
    for is_ARD in (false, true)
        for T in FloatingPointTypes
            case_args = T[test_args...]
            x, y, w = T[1], T[2], T[2]
            K = is_ARD ? ARD((kernelobject)(case_args...),w) : (kernelobject)(case_args...)
            z = is_ARD ? sum(x .* y .* w.^2) : sum(x .* y)
            test_value = test_function(z, case_args...)
            print("    - Testing ", K, "... ")
            @test_approx_eq_type MLKernels.kappa((is_ARD ? K.k : K), z) test_value T
            test_kernel_function(K, x, y, test_value)
            test_kernel_function(K, x[1], y[1], test_value)
            println("Done")
        end
    end
end

println("- Testing SquaredDistanceKernel kernel() function:")
for (kernelobject, test_args, test_function) in (
        (ExponentialKernel, [2,0.5], (z,a,t) -> exp(-a * z^t)),
        (RationalQuadraticKernel, [2,2,0.5], (z,a,b,t) -> (1 + a*z^t)^(-b)),
        (PowerKernel, [0.5], (z,t) -> -z^t),
        (LogKernel, [2,0.5], (z,a,t) -> -log(a*z^t + 1)),
        (MaternKernel, [2,2], (z,a,t) -> 2(sqrt(2a*z)/2t)^a * besselk(a,z)/gamma(a)),
    )
    for is_ARD in (false,true)
        for T in FloatingPointTypes
            case_args = T[test_args...]
            x, y, w = T[1], T[2], T[2]
            K = is_ARD ? ARD((kernelobject)(case_args...), w) : (kernelobject)(case_args...)
            lag = is_ARD ? w .* (x - y) : x - y
            z = dot(lag,lag)
            test_value = test_function(z, case_args...)
            print("    - Testing ", K, "... ")
            @test_approx_eq_type MLKernels.kappa((is_ARD ? K.k : K), z) test_value T
            test_kernel_function(K, x, y, test_value)
            test_kernel_function(K, x[1], y[1], test_value)
            println("Done")
        end
    end
end

println("- Testing special case kappa() function:")
for (kernelobject, test_args_set, test_function) in (
        (ExponentialKernel, ([1,1],), (z,a,t) -> exp(-a * z^t)),
        (RationalQuadraticKernel, ([2,1,0.5],[2,2,1],[2,1,1],), (z,a,b,t) -> (1 + a*z^t)^(-b)),
        (PowerKernel, ([1],), (z,t) -> -z^t),
        (LogKernel, ([2,1],), (z,a,t) -> -log(a*z^t + 1)),
        (MaternKernel, ([1,1],), (z,a,t) -> 2(sqrt(2a*z)/2t)^a * besselk(a,z)/gamma(a)),
        (PolynomialKernel, ([1,1,1],), (z,a,c,d) -> (a * z + c )^d),
    )
    for T in FloatingPointTypes
        for test_args in test_args_set
            case_args = T[test_args...]
            Knil = kernelobject{T,:nil}(case_args...)
            Kcase = (kernelobject)(case_args...)
            print("    - Testing ", typeof(Kcase), "... ")
            for z = T[0, 0.1, 0.5, 1.0, 2.0]
                test_value = test_function(z, case_args...)
                @test_approx_eq_type MLKernels.kappa(Kcase, z) test_value T
                @test_approx_eq_type MLKernels.kappa(Knil, z) test_value T
            end
            println("Done")
        end
    end
end

println("- Testing special case error cases:")
for (kernelobject, case, test_args) in (
        (ExponentialKernel, :γ1, [1,0.5]),
        (RationalQuadraticKernel, :β1, [2,2,1]),
        (RationalQuadraticKernel, :γ1, [2,1,0.5]),
        (RationalQuadraticKernel, :β1γ1, [2,2,0.5]),
        (PowerKernel, :γ1, [0.5]),
        (LogKernel, :γ1, [2,0.5]),
        (MaternKernel, :ν1, [0.5,1]),
        (PolynomialKernel, :d1, [1,1,2]),
    )
    for T in FloatingPointTypes
        case_args = T[test_args...]
        print("    - Testing ", kernelobject{T,case}, "... ")
        @test_throws ErrorException kernelobject{T,case}(case_args...)
        println("Done")
    end
end

# Test KernelProduct
print("- Testing KernelProduct constructors ... ")
for T in FloatingPointTypes
    x, y, a, b = ([one(T)], [one(T)], convert(T,2), convert(T,3))

    K1 = ExponentialKernel(one(T))
    K2 = RationalQuadraticKernel(one(T))
    K3 = PolynomialKernel(one(T))
    K4 = SigmoidKernel(one(T))

    for (K1K2, m, n) in (
            (KernelProduct(a, K1, K2), a, 2),
            (K1 * K2,                  one(T), 2), # Kernel,Kernel
            (a * K1,                   a,      1), # Real,Kernel
            (K1 * a,                   a,      1), # Kernel,Real
            (a * (K1*K2*b),            a*b,    2), # Real,KernelProduct
            ((K1*K2*b) * a,            a*b,    2), # KernelProduct,Real
            ((a*K1) * (K2*K3*b),       a*b,    3), # KernelProduct,KernelProduct
            (K1 * (a*K2*K3),           a,      3), # Kernel,KernelProduct
            ((a*K1*K2) * K3,           a,      3), # KernelProduct,Kernel
        )

        @test K1K2.a == m
        @test typeof(K1K2.a) == T

        check_fields(K1K2.k[1], K1)
        n >= 2 && check_fields(K1K2.k[2], K2)
        n >= 3 && check_fields(K1K2.k[3], K3)
    end

    @test ismercer(K1*K2*K3) == true
    @test iscondposdef(K1*K2*K3*K4) == false

end
println(" Done")

# Test KernelSum
print("- Testing KernelSum constructors ... ")
for T in FloatingPointTypes
    x, y = ([one(T)], [one(T)])

    K1 = ExponentialKernel(one(T))
    K2 = RationalQuadraticKernel(one(T))
    K3 = PolynomialKernel(one(T))
    K4 = SigmoidKernel(one(T))

    for (K1K2, n) in (
            (KernelSum(one(T),K1, K2),     2),
            (K1 + K2,               2),
            (K1 + (K2 + K3),        3),
            ((K1 + K2) + K3,        3),
            ((K1 + K2) + (K3 + K4), 4),
        )
        n >= 1 && check_fields(K1K2.k[1], K1)
        n >= 2 && check_fields(K1K2.k[2], K2)
        n >= 3 && check_fields(K1K2.k[3], K3)
        n >= 4 && check_fields(K1K2.k[4], K4)
        show(STDOUT, K1K2)
    end

    @test ismercer(K1+K2+K3) == true
    @test iscondposdef(K1+K2+K3+K4) == false

end
println(" Done")

print("- Testing CompositeKernel kernel function ... ")
for T in FloatingPointTypes
    x, y = T[1, 1], T[2, 3]
    x1, y1 = T[1], T[2]
    a = convert(T, 3)

    K1 = ExponentialKernel(one(T))
    K2 = RationalQuadraticKernel(one(T))
    K3 = PolynomialKernel(one(T))
    K4 = SigmoidKernel(one(T))

    kernels = [a, K1, K2, K3, K4]
    values = [a, map(k -> kernel(k, x, y), kernels[2:end])...]
    val1 = [a, map(k -> kernel(k, x1, y1), kernels[2:end])...]
    for op in (prod, sum)
        test_kernel_function(op(kernels), x, y, op(values))
        test_kernel_function(op(kernels), x1, y1, op(val1))
    end
    K = a * (K1 + K2) * (K3 + K4 + a)
    v = a * (kernel(K1, x, y) + kernel(K2, x, y)) * (kernel(K3, x, y) + kernel(K4, x, y) + a)
    v1 = a * (kernel(K1, x1, y1) + kernel(K2, x1, y1)) * (kernel(K3, x1, y1) + kernel(K4, x1, y1) + a)
    show(STDOUT, K)
    test_kernel_function(K, x, y, v)
    test_kernel_function(K, x1, y1, v1)
end
println(" Done")

=#
