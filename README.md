# KernelFunctions.jl

KernelFunctions.jl is a Julia package for (Mercer) kernel functions used in kernel methods in the
kernel methods of machine learning.

## Using Kernel Functions

A number of standard kernels have been pre-defined. For example, to create a Polynomial Kernel object:

```julia
julia> using KernelFunctions

julia> κ = PolynomialKernel(1,0,2)
 PolynomialKernel(α=1,c=0,d=2)
```

If one wishes to see more information on the kernel, there is the function `description` which will print out a description of the kernel if it exists:

```julia
julia> description(κ)
 Polynomial Kernel:
 
 The polynomial kernel is a non-stationary kernel which represents
 the original features as in a feature space over polynomials up to 
 degree d of the original variables:

     k(x,y) = (αxᵗy + c)ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0, d > 0

 This kernel is sensitive to numerical instability in the case that
 d is increasingly large and αxᵗy + c approaches zero.
```

To utilise the kernel function for the associated kernel, one simply uses the `kernel_function` method with the first argument being the kernel of choice:

```julia
julia> x, y = ([1.0, 2.0], [1.0, 1.0])
([1.0,2.0],[1.0,1.0])

julia> kernel_function(κ, x, y)
16.0
```

Note that the kernels are callable objects ("functors"), so the same as above may be accomplished through:

```julia
julia> PolynomialKernel()(x, y)
16.0

julia> κ(x, y)
16.0
```

Not all of the provided kernels are positive-definite. The function `isposdef_kernel` will return `true` if the kernel is positive definite, and `false` otherwise.

```julia
julia> isposdef_kernel(κ)
true
```

Given a data matrix with one observation per row, thekernel matrix for kernel `κ` can be calculated using the `kernel_matrix` method:

```julia
julia> X = rand(10, 5)
10x5 Array{Float64,2}:
 0.504367  0.00359014  0.33527   0.525783   0.309577 
 0.743592  0.00457371  0.594875  0.336426   0.102312 
 0.713523  0.84033     0.109831  0.0415291  0.830992 
 0.311136  0.312127    0.441631  0.484347   0.600404 
 0.756499  0.307774    0.473343  0.66326    0.63204  
 0.154361  0.174706    0.467842  0.504841   0.0189265
 0.72457   0.683054    0.675841  0.79224    0.252396 
 0.148947  0.704279    0.472177  0.263061   0.016687 
 0.953466  0.286904    0.386647  0.150966   0.863391 
 0.736272  0.497723    0.910921  0.58754    0.666643 

julia> kernel_matrix(κ, X)
10x10 Array{Float64,2}:
 3.02444  3.17932  2.8184   3.05077  …  4.36464  1.90285  3.83462   4.81293
 3.17932  4.12283  2.88572  2.95797     5.00105  2.20544  4.32404   5.55788
 2.8184   2.88572  8.52409  4.21006     5.79777  3.14976  7.2233    6.875  
 3.05077  2.95797  4.21006  3.93762     5.16367  2.59895  4.61591   6.10857
 4.35036  4.55104  5.7752   5.02302     7.63557  3.02052  6.96137   8.71661
 2.26994  2.45131  1.8091   2.44939  …  3.81068  2.25017  2.16309   3.74812
 4.36464  5.00105  5.79777  5.16367     9.85755  4.49745  6.17847   9.75201
 1.90285  2.20544  3.14976  2.59895     4.49745  3.27836  2.49881   4.22714
 3.83462  4.32404  7.2233   4.61591     6.17847  2.49881  8.46311   8.18696
 4.81293  5.55788  6.875    6.10857     9.75201  4.22714  8.18696  11.6228 

julia> kernel_function(κ, vec(X[1,:]), vec(X[1,:]))
3.02443612827351
```
