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
 ===================================================================
 The polynomial kernel is a non-stationary kernel which represents
 the original features as in a feature space over polynomials up to 
 degree d of the original variables:

     k(x,y) = (αxᵗy + c)ᵈ    x ∈ ℝⁿ, y ∈ ℝⁿ, α > 0, c ≥ 0, d > 0

 This kernel is sensitive to numerical instability in the case that
 d is increasingly large and αxᵗy + c approaches zero.
```

To extract the associated kernel, one may use the 'kernel_function' method:

```julia
julia> k = kernel_function(κ)
k (generic function with 1 method)

julia> x, y = ([1.0, 2.0], [1.0, 1.0])
([1.0,2.0],[1.0,1.0])

julia> k(x,y)
9.0
```

Note that the kernels are callable objects ("functors"), so the same as above may be accomplished through:

```julia
julia> PolynomialKernel(1,0,2)(x,y)
9.0

julia> κ(x,y)
9.0
```


