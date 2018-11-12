var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Machine-Learning-Kernels-1",
    "page": "Home",
    "title": "Machine Learning Kernels",
    "category": "section",
    "text": "MLKernels.jl is a Julia package for Mercer  kernel functions (or the covariance functions of Gaussian processes) that are used in the  kernel methods of machine learning. This package provides a collection of kernel datatypes  for representing kernel functions as well as an efficient set of methods to compute or  approximate kernel matrices. "
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "The package may be added by running one of the following code:(v0.7) pkg> add MLKernels"
},

{
    "location": "#Getting-Started-1",
    "page": "Home",
    "title": "Getting Started",
    "category": "section",
    "text": "Documentation on the interface implemented by MLKernels.jl is available under the  interface section listed in the table of contents below or on the sidebar:Interface - the MLKernels API\nKernels - a listing of the implemented kernel functions and their properties\nKernel Theory - documentation on the theory surrounding kernel functions and the kernel trick"
},

{
    "location": "interface/#",
    "page": "Interface",
    "title": "Interface",
    "category": "page",
    "text": ""
},

{
    "location": "interface/#Interface-1",
    "page": "Interface",
    "title": "Interface",
    "category": "section",
    "text": ""
},

{
    "location": "interface/#Data-Orientation-1",
    "page": "Interface",
    "title": "Data Orientation",
    "category": "section",
    "text": "Data matrices may be oriented in one of two ways with respect to the observations. Functions producing a kernel matrix require an orient argument to specify the orientation of the observations within the provided data matrix."
},

{
    "location": "interface/#Row-Orientation-(Default)-1",
    "page": "Interface",
    "title": "Row Orientation (Default)",
    "category": "section",
    "text": "An orientation of Val(:row) identifies when observation vector corresponds to a row of the data matrix. This is commonly used in the field of statistics in the context of design matrices.For example, for data matrix mathbfX consisting of observations mathbfx_1, mathbfx_2, ldots, mathbfx_n:mathbfX_row =\nbeginbmatrix\n    leftarrow mathbfx_1 rightarrow \n    leftarrow mathbfx_2 rightarrow \n    vdots \n    leftarrow mathbfx_n rightarrow\nendbmatrixWhen row-major ordering is used, then the kernel matrix of mathbfX will match the dimensions of mathbfX^intercalmathbfX. Similarly, the kernel matrix will match the dimension of mathbfX^intercalmathbfY for row-major ordering of data matrix mathbfX and mathbfY."
},

{
    "location": "interface/#Column-Orientation-1",
    "page": "Interface",
    "title": "Column Orientation",
    "category": "section",
    "text": "An orientation of Val(:col) identifies when each observation vector corresponds to a column of the data matrix:mathbfX_col =\nmathbfX_row^intercal =\nbeginbmatrix\n    uparrow  uparrow   uparrow  \n    mathbfx_1  mathbfx_2  cdots  mathbfx_n \n    downarrow  downarrow   downarrow\nendbmatrixWith column-major ordering, the kernel matrix will match the dimensions of mathbfXX^intercal. Similarly, the kernel matrix of data matrices mathbfX and mathbfY match the dimensions of mathbfXY^intercal."
},

{
    "location": "interface/#MLKernels.ismercer-Tuple{Kernel}",
    "page": "Interface",
    "title": "MLKernels.ismercer",
    "category": "method",
    "text": "ismercer(κ::Kernel)\n\nReturns true if kernel κ is a Mercer kernel; false otherwise.\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.isnegdef-Tuple{Kernel}",
    "page": "Interface",
    "title": "MLKernels.isnegdef",
    "category": "method",
    "text": "isnegdef(κ::Kernel)\n\nReturns true if the kernel κ is a negative definite kernel; false otherwise.\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.isstationary-Tuple{Kernel}",
    "page": "Interface",
    "title": "MLKernels.isstationary",
    "category": "method",
    "text": "isstationary(κ::Kernel)\n\nReturns true if the kernel κ is a stationary kernel; false otherwise.\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.isisotropic-Tuple{Kernel}",
    "page": "Interface",
    "title": "MLKernels.isisotropic",
    "category": "method",
    "text": "isisotropic(κ::Kernel)\n\nReturns true if the kernel κ is an isotropic kernel; false otherwise.\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.kernel-Union{Tuple{T}, Tuple{Kernel{T},Real,Real}} where T",
    "page": "Interface",
    "title": "MLKernels.kernel",
    "category": "method",
    "text": "kernel(κ::Kernel, x, y)\n\nApply the kernel κ to x and y where x and y are vectors or scalars of some subtype of Real.\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.Orientation",
    "page": "Interface",
    "title": "MLKernels.Orientation",
    "category": "constant",
    "text": "Orientation\n\nUnion of the two Val types representing the data matrix orientations:\n\nVal{:row} identifies when observation vector corresponds to a row of the data matrix\nVal{:col} identifies when each observation vector corresponds to a column of the data matrix\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.kernelmatrix-Union{Tuple{T1}, Tuple{T}, Tuple{Union{Val{:row}, Val{:col}},Kernel{T},AbstractArray{T1,2},Bool}} where T1 where T",
    "page": "Interface",
    "title": "MLKernels.kernelmatrix",
    "category": "method",
    "text": "kernelmatrix([σ::Orientation,] κ::Kernel, X::Matrix [, symmetrize::Bool])\n\nCalculate the kernel matrix of X with respect to kernel κ.\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.kernelmatrix!-Union{Tuple{T}, Tuple{Union{Val{:row}, Val{:col}},Array{T,2},Kernel{T},AbstractArray{T,2},Bool}} where T<:AbstractFloat",
    "page": "Interface",
    "title": "MLKernels.kernelmatrix!",
    "category": "method",
    "text": "kernelmatrix!(P::Matrix, σ::Orientation, κ::Kernel, X::Matrix, symmetrize::Bool)\n\nIn-place version of kernelmatrix where pre-allocated matrix K will be overwritten with the kernel matrix.\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.kernelmatrix-Union{Tuple{T2}, Tuple{T1}, Tuple{T}, Tuple{Union{Val{:row}, Val{:col}},Kernel{T},AbstractArray{T1,2},AbstractArray{T2,2}}} where T2 where T1 where T",
    "page": "Interface",
    "title": "MLKernels.kernelmatrix",
    "category": "method",
    "text": "kernelmatrix([σ::Orientation,] κ::Kernel, X::Matrix, Y::Matrix)\n\nCalculate the base matrix of X and Y with respect to kernel κ.\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.kernelmatrix!-Union{Tuple{T}, Tuple{Union{Val{:row}, Val{:col}},Array{T,2},Kernel{T},AbstractArray{T,2},AbstractArray{T,2}}} where T<:AbstractFloat",
    "page": "Interface",
    "title": "MLKernels.kernelmatrix!",
    "category": "method",
    "text": "kernelmatrix!(K::Matrix, σ::Orientation, κ::Kernel, X::Matrix, Y::Matrix)\n\nIn-place version of kernelmatrix where pre-allocated matrix K will be overwritten with the kernel matrix.\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.centerkernelmatrix!-Union{Tuple{Array{T,2}}, Tuple{T}} where T<:AbstractFloat",
    "page": "Interface",
    "title": "MLKernels.centerkernelmatrix!",
    "category": "method",
    "text": "centerkernelmatrix(K::Matrix)\n\nCenters the (rectangular) kernel matrix K with respect to the implicit Kernel Hilbert Space according to the following formula:\n\nmathbfK_ij\n= langlephi(mathbfx_i) -mathbfmu_phimathbfx phi(mathbfy_j)\n- mathbfmu_phimathbfy rangle\n\nWhere mathbfmu_phimathbfx and mathbfmu_phimathbfx are given by:\n\nmathbfmu_phimathbfx\n= frac1n sum_i=1^n phi(mathbfx_i)\nqquad qquad\nmathbfmu_phimathbfy\n= frac1m sum_i=1^m phi(mathbfy_i)\n\n\n\n\n\n"
},

{
    "location": "interface/#Essentials-1",
    "page": "Interface",
    "title": "Essentials",
    "category": "section",
    "text": "ismercer(::Kernel)\nisnegdef(::Kernel)\nisstationary(::Kernel)\nisisotropic(::Kernel)\nkernel(::Kernel{T}, ::Real, ::Real) where T\nOrientation\nkernelmatrix(::Orientation, ::Kernel{T}, ::AbstractMatrix{T1}, symmetrize::Bool) where {T,T1}\nkernelmatrix!(::Orientation, ::Matrix{T}, ::Kernel{T}, ::AbstractMatrix{T}, symmetrize::Bool) where {T<:AbstractFloat}\nkernelmatrix(::Orientation, ::Kernel{T}, ::AbstractMatrix{T1}, ::AbstractMatrix{T2}) where {T,T1,T2}\nkernelmatrix!(::Orientation, ::Matrix{T}, ::Kernel{T}, ::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T<:AbstractFloat}\ncenterkernelmatrix!(::Matrix{T}) where {T<:AbstractFloat}"
},

{
    "location": "interface/#MLKernels.NystromFact",
    "page": "Interface",
    "title": "MLKernels.NystromFact",
    "category": "type",
    "text": "NystromFact\n\nType for storing a Nystrom factorization. The factorization contains two fields: W and C as described in the nystrom documentation.\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.nystrom",
    "page": "Interface",
    "title": "MLKernels.nystrom",
    "category": "function",
    "text": "nystrom([σ::Orientation,] κ::Kernel, X::Matrix, [S::Vector])\n\nComputes a factorization of Nystrom approximation of the square kernel matrix of data matrix X with respect to kernel κ. Returns a NystromFact struct which stores a Nystrom factorization satisfying:\n\nmathbfK approx mathbfC^intercalmathbfWC\n\n\n\n\n\n"
},

{
    "location": "interface/#MLKernels.kernelmatrix-Union{Tuple{NystromFact{T}}, Tuple{T}} where T<:Union{Float32, Float64}",
    "page": "Interface",
    "title": "MLKernels.kernelmatrix",
    "category": "method",
    "text": "nystrom(CᵀWC::NystromFact)\n\nCompute the approximate kernel matrix based on the Nystrom factorization.\n\n\n\n\n\n"
},

{
    "location": "interface/#Approximation-1",
    "page": "Interface",
    "title": "Approximation",
    "category": "section",
    "text": "In many cases, fast, approximate results is more important than a perfect result. The Nystrom method can be used to generate a factorization that can be used to approximate a large, symmetric kernel matrix. Given data matrix mathbfX in mathbbR^n times p (one observation per row) and kernel matrix mathbfK in mathbbR^n times n, the Nystrom method takes a sample S of the observations of mathbfX of size s  n and generates a factorization such that:mathbfK approx mathbfC^intercalmathbfWCWhere mathbfW is the s times s pseudo-inverse of the sample kernel matrix based on S and mathbfC is a s times n matrix.The Nystrom method uses an eigendecomposition of the sample kernel matrix of mathbfX to estimate mathbfK. Generally, the order of mathbfK must be quite large and the sampling ratio small (ex. 15% or less) for the cost of the computing the full kernel matrix to exceed that of the eigendecomposition. This method will be more effective for kernels that are not a direct function of the dot product as they are not able to make use of BLAS in computing the full matrix mathbfK and the cross-over point will occur for smaller mathbfK.MLKernels.jl implements the Nystrom approximation:NystromFact\nnystrom\nkernelmatrix(::NystromFact{T}) where {T<:LinearAlgebra.BlasReal}"
},

{
    "location": "kernels/#",
    "page": "Kernels",
    "title": "Kernels",
    "category": "page",
    "text": ""
},

{
    "location": "kernels/#Kernels-1",
    "page": "Kernels",
    "title": "Kernels",
    "category": "section",
    "text": ""
},

{
    "location": "kernels/#MLKernels.ExponentialKernel",
    "page": "Kernels",
    "title": "MLKernels.ExponentialKernel",
    "category": "type",
    "text": "ExponentialKernel([α=1])\n\nThe exponential kernel is given by the formula:\n\nkappa(mathbfxmathbfy) = expleft(-alpha mathbfx - mathbfyright) \nqquad alpha  0\n\nwhere alpha is a scaling parameter of the Euclidean distance. The exponential kernel,  also known as the Laplacian kernel, is an isotropic Mercer kernel. The constructor is  aliased by LaplacianKernel, so both names may be used:\n\n\n\n\n\n"
},

{
    "location": "kernels/#MLKernels.SquaredExponentialKernel",
    "page": "Kernels",
    "title": "MLKernels.SquaredExponentialKernel",
    "category": "type",
    "text": "SquaredExponentialKernel([α=1])\n\nThe squared exponential kernel, or alternatively the Gaussian kernel, is identical to the  exponential kernel except that the Euclidean distance is squared:\n\nkappa(mathbfxmathbfy) = expleft(-alpha mathbfx - mathbfy^2right) \nqquad alpha  0\n\nwhere alpha is a scaling parameter of the squared Euclidean distance. Just like the exponential kernel, the squared exponential kernel is an isotropic Mercer kernel. The squared exponential kernel is more commonly known as the radial basis kernel within machine learning communities.\n\n\n\n\n\n"
},

{
    "location": "kernels/#MLKernels.GammaExponentialKernel",
    "page": "Kernels",
    "title": "MLKernels.GammaExponentialKernel",
    "category": "type",
    "text": "GammaExponentialKernel([α=1 [,γ=1]])\n\nThe gamma exponential kernel is a generalization of the exponential and squared exponential  kernels:\n\nkappa(mathbfxmathbfy) = expleft(-alpha mathbfx - mathbfy^gamma \nright) qquad alpha  0  0  gamma leq 1\n\nwhere alpha is a scaling parameter and gamma is a shape parameter.\n\n\n\n\n\n"
},

{
    "location": "kernels/#MLKernels.RationalQuadraticKernel",
    "page": "Kernels",
    "title": "MLKernels.RationalQuadraticKernel",
    "category": "type",
    "text": "RationalQuadraticKernel([α=1 [,β=1]])\n\nThe rational-quadratic kernel is given by:\n\nkappa(mathbfxmathbfy) \n= left(1 +alpha mathbfxmathbfy^2right)^-beta \nqquad alpha  0  beta  0\n\nwhere alpha is a scaling parameter and beta is a shape parameter. This kernel can  be seen as an infinite sum of Gaussian kernels. If one sets alpha = alpha_0  beta,  then taking the limit beta rightarrow infty results in the Gaussian kernel with  scaling parameter alpha_0.\n\n\n\n\n\n"
},

{
    "location": "kernels/#MLKernels.GammaRationalKernel",
    "page": "Kernels",
    "title": "MLKernels.GammaRationalKernel",
    "category": "type",
    "text": "GammaRationalKernel([α [,β [,γ]]])\n\nThe gamma-rational kernel is a generalization of the rational-quadratic kernel with an  additional shape parameter:\n\nkappa(mathbfxmathbfy)\n= left(1 +alpha mathbfxmathbfy^gammaright)^-beta \nqquad alpha  0  beta  0  0  gamma leq 1\n\nwhere alpha is a scaling parameter and beta and gamma are shape parameters.\n\n\n\n\n\n"
},

{
    "location": "kernels/#MLKernels.MaternKernel",
    "page": "Kernels",
    "title": "MLKernels.MaternKernel",
    "category": "type",
    "text": "MaternKernel([ν=1 [,θ=1]])\n\nThe Matern kernel is a Mercer kernel given by:\n\nkappa(mathbfxmathbfy) =\nfrac12^nu-1Gamma(nu)\nleft(fracsqrt2numathbfx-mathbfythetaright)^nu\nK_nuleft(fracsqrt2numathbfx-mathbfythetaright)\n\nwhere Gamma is the gamma function, K_nu is the modified Bessel function of the second kind, nu  0 and theta  0.\n\n\n\n\n\n"
},

{
    "location": "kernels/#MLKernels.LinearKernel",
    "page": "Kernels",
    "title": "MLKernels.LinearKernel",
    "category": "type",
    "text": "LinearKernel([a=1 [,c=1]])\n\nThe linear kernel is a Mercer kernel given by:\n\nkappa(mathbfxmathbfy) = \na mathbfx^intercal mathbfy + c qquad alpha  0  c geq 0\n\n\n\n\n\n"
},

{
    "location": "kernels/#MLKernels.PolynomialKernel",
    "page": "Kernels",
    "title": "MLKernels.PolynomialKernel",
    "category": "type",
    "text": "PolynomialKernel([a=1 [,c=1 [,d=3]]])\n\nThe polynomial kernel is a Mercer kernel given by:\n\nkappa(mathbfxmathbfy) = \n(a mathbfx^intercal mathbfy + c)^d\nqquad alpha  0  c geq 0  d in mathbbZ_+\n\n\n\n\n\n"
},

{
    "location": "kernels/#MLKernels.ExponentiatedKernel",
    "page": "Kernels",
    "title": "MLKernels.ExponentiatedKernel",
    "category": "type",
    "text": "ExponentiatedKernel([a=1])\n\nThe exponentiated kernel is a Mercer kernel given by:\n\nkappa(mathbfxmathbfy) = expleft(a mathbfx^intercal mathbfy right) \nqquad a  0\n\n\n\n\n\n"
},

{
    "location": "kernels/#MLKernels.PeriodicKernel",
    "page": "Kernels",
    "title": "MLKernels.PeriodicKernel",
    "category": "type",
    "text": "PeriodicKernel([α=1 [,p=π]])\n\nThe periodic kernel is given by:\n\nkappa(mathbfxmathbfy) =\nexpleft(-alpha sum_i=1^n sin(p(x_i - y_i))^2right)\nqquad p 0  alpha  0\n\nwhere mathbfx and mathbfy are n dimensional vectors. The parameters p  and alpha are scaling parameters for the periodicity and the magnitude, respectively.  This kernel is useful when data has periodicity to it.\n\n\n\n\n\n"
},

{
    "location": "kernels/#Mercer-Kernels-1",
    "page": "Kernels",
    "title": "Mercer Kernels",
    "category": "section",
    "text": "ExponentialKernel\nSquaredExponentialKernel\nGammaExponentialKernel\nRationalQuadraticKernel\nGammaRationalKernel\nMaternKernel\nLinearKernel\nPolynomialKernel\nExponentiatedKernel\nPeriodicKernel"
},

{
    "location": "kernels/#MLKernels.PowerKernel",
    "page": "Kernels",
    "title": "MLKernels.PowerKernel",
    "category": "type",
    "text": "PowerKernel([γ=1])\n\nThe Power Kernel is a negative definite kernel given by:\n\nkappa(mathbfxmathbfy) = \nmathbfx - mathbfy ^2gamma\nqquad gamma in (01\n\n\n\n\n\n"
},

{
    "location": "kernels/#MLKernels.LogKernel",
    "page": "Kernels",
    "title": "MLKernels.LogKernel",
    "category": "type",
    "text": "LogKernel([α [,γ]])\n\nThe Log Kernel is a negative definite kernel given by:\n\nkappa(mathbfxmathbfy) = \nlog left(1 + alphamathbfx - mathbfy ^2gammaright)\nqquad alpha  0  gamma in (01\n\n\n\n\n\n"
},

{
    "location": "kernels/#Negative-Definite-Kernels-1",
    "page": "Kernels",
    "title": "Negative Definite Kernels",
    "category": "section",
    "text": "PowerKernel\nLogKernel"
},

{
    "location": "kernels/#MLKernels.SigmoidKernel",
    "page": "Kernels",
    "title": "MLKernels.SigmoidKernel",
    "category": "type",
    "text": "SigmoidKernel([a=1 [,c=1]])\n\nThe Sigmoid Kernel is given by\n\nkappa(mathbfxmathbfy) = \ntanh(a mathbfx^intercal mathbfy + c) \nqquad alpha  0  c geq 0\n\nThe sigmoid kernel is a not a true kernel, although it has been used in application. \n\n\n\n\n\n"
},

{
    "location": "kernels/#Other-Kernels-1",
    "page": "Kernels",
    "title": "Other Kernels",
    "category": "section",
    "text": "SigmoidKernel"
},

{
    "location": "kernel-theory/#",
    "page": "Kernel Theory",
    "title": "Kernel Theory",
    "category": "page",
    "text": ""
},

{
    "location": "kernel-theory/#Kernel-Theory-1",
    "page": "Kernel Theory",
    "title": "Kernel Theory",
    "category": "section",
    "text": ""
},

{
    "location": "kernel-theory/#The-Kernel-Trick-1",
    "page": "Kernel Theory",
    "title": "The Kernel Trick",
    "category": "section",
    "text": "Many machine and statistical learning algorithms, such as support vector machines and  principal components analysis, are based on inner products. These methods can often be  generalized through use of the kernel trick to create anonlinear decision boundary  without using an explicit mapping to another space. The kernel trick makes use of Mercer kernels which operate on vectors in the input  space but can be expressed as inner products in another space. In other words, if  mathcalX is the input vector space and kappa is the Mercer kernel function,  then for some vector space mathcalV there exists a function \\phi such that:kappa(x_1 x_2) \n= leftlangle phi(x_1) phi(x_2)rightrangle_mathcalV\nqquad x_1 x_2 in mathcalXIn machine learning, the vector space mathcalX is known as the feature space and the  function phi is known as a feature map. A simple example of a feature map can be shown  with the Polynomial Kernel:kappa(mathbfxmathbfy) = (amathbfx^intercalmathbfy + c)^d\nqquad mathbfxmathbfy in mathbbR^n \nquad a c in mathbbR_+\nquad d in mathbbZ_+In our example, we will use n=2, d=2, a=1 and c=0. Substituting these  values in, we get the following kernel function:kappa(mathbfxmathbfy) = left(x_1 y_1 + x_2 y_2right)^2\n= x_1^2 y_1^2 + x_1 x_2 y_1 y_2 + x_2^2 y_2^2\n= phi(mathbfx)^intercalphi(mathbfy)Where the feature map phi  mathbbR^2 rightarrow mathbbR^3 is defined by:phi(mathbfx) = \nbeginbmatrix\n    x_1^2 \n    x_1 x_2 \n    x_2^2\nendbmatrixThe advantage of the implicit feature map is that we may transform non-linearly data into  linearly separable data in the implicit space."
},

{
    "location": "kernel-theory/#Kernels-1",
    "page": "Kernel Theory",
    "title": "Kernels",
    "category": "section",
    "text": "The kernel methods are a class of algorithms that are used for pattern analysis. These  methods make use of kernel functions. A symmetric, real valued kernel function  kappa mathcalX times mathcalX rightarrow mathbbR is said to be positive  definite or Mercer if and only:sum_i=1^n sum_j=1^n c_i c_j kappa(mathbfx_imathbfx_j) geq 0for all n in mathbbN, mathbfx_1 dots mathbfx_n subseteq mathcalX and c_1 dots c_n subseteq mathbbR. Similarly, a real valued kernel function is said to be negative definite if and only if:sum_i=1^n sum_j=1^n c_i c_j kappa(mathbfx_imathbfx_j) leq 0 qquad sum_i=1^n c_i = 0for n geq 2, mathbfx_1 dots mathbfx_n subseteq mathcalX and  c_1 dots c_n subseteq mathbbR. In machine learning literature,  conditionally positive definite kernels are often studied instead. This is simply a  reversal of the above inequality. Trivially, every negative definite kernel can be  transformed into a conditionally positive definite kernel by negation."
},

{
    "location": "kernel-theory/#Further-Reading-1",
    "page": "Kernel Theory",
    "title": "Further Reading",
    "category": "section",
    "text": "Berg C, Christensen JPR, Ressel P. 1984. Harmonic Analysis on Semigroups. Springer-Verlag New York. Chapter 3, General Results on Positive and Negative Definite Matrices and Kernels; p. 66-85.\nBouboulis P. 2014. Academic Press Library in Signal Processing, Volume 1: Array and Statistical Signal Processing (1st ed.). Academic Press. Chapter 17, Online Learning in Reproducing Kernel Hilbert Spaces; p. 883-987.\nGenton M.G. 2002. Classes of kernels for machine learning: a statistics perspective. The Journal of Machine Learning Research. Volume 2 (March 2002), 299-312.\nRasmussen C, Williams CKI. 2005. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press. Chapter 4, Covariance Functions; p. 79-104."
},

]}
