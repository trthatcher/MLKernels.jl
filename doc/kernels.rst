----------------
Kernel Functions
----------------


Exponential Kernel
..................

.. class:: ExponentialKernel([α::Real=1]) <: MercerKernel

  The exponential kernel is given by the formula:

  .. math::

      \kappa(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||\right) \qquad \alpha > 0

  where :math:`\alpha` is a scaling parameter of the Euclidean distance. The
  exponential kernel, also known as the Laplacian kernel, is an isotropic Mercer 
  kernel. The constructor is aliased by ``LaplacianKernel``, so both names may 
  be used:

  .. code-block:: julia

      ExponentialKernel()  # Default is Float64 with α = 1.0
      LaplacianKernel(1)   # Integers will be converted to Float64


Squared Exponential Kernel
..........................

.. class:: SquaredExponentialKernel([α::Real=1]) <: MercerKernel

  The squared exponential kernel, or alternatively the Gaussian kernel, is 
  identical to the exponential kernel except that the Euclidean distance is 
  squared:

  .. math::

      \kappa(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^2\right) \qquad \alpha > 0

  where :math:`\alpha` is a scaling parameter of the squared Euclidean distance.
  Just like the exponential kernel, the squared exponential kernel is an
  isotropic Mercer kernel. The squared exponential kernel is more commonly known
  as the radial basis kernel within machine learning communities. All aliases
  may be used in MLKernels.jl:

  .. code-block:: julia

    GaussianKernel()
    RadialBasisKernel()
    SquaredExponentialKernel()


Gamma Exponential Kernel
........................

.. class:: GammaExponentialKernel([α::Real=1 [,γ::Real=1]]) <: MercerKernel

  The gamma exponential kernel is a generalization of the exponential and
  squared exponential kernels:

  .. math::

    \kappa(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha ||\mathbf{x} - \mathbf{y}||^{\gamma}\right) \qquad \alpha > 0, \; 0 < \gamma \leq 1

  where :math:`\alpha` is a scaling parameter and :math:`\gamma` is a shape
  parameter.


Rational-Quadratic Kernel
.........................

.. class:: RationalQuadraticKernel([α::Real=1 [,β::Real=1]]) <: MercerKernel

  The rational-quadratic kernel is given by:

  .. math::

      \kappa(\mathbf{x},\mathbf{y}) = \left(1 +\alpha ||\mathbf{x},\mathbf{y}||^2\right)^{-\beta} \qquad \alpha > 0, \; \beta > 0

  where :math:`\alpha` is a scaling parameter and :math:`\beta` is a shape
  parameter. This kernel can be seen as an infinite sum of Gaussian kernels. If
  one sets :math:`\alpha = \alpha_0 / \beta`, then taking the limit :math:`\beta
  \rightarrow \infty` results in the Gaussian kernel with scaling parameter
  :math:`\alpha_0`. 


Gamma-Rational Kernel
.....................

.. class:: RationalQuadraticClass([α::Real [,β::Real [,γ::Real]]]) <: MercerKernel
  
  The gamma-rational kernel is a generalization of the rational-quadratic kernel
  with an additional shape parameter:

  .. math::

    \kappa(\mathbf{x},\mathbf{y}) = \left(1 +\alpha ||\mathbf{x},\mathbf{y}||^{\gamma}\right)^{-\beta} \qquad \alpha > 0, \; \beta > 0, \; 0 < \gamma \leq 1

  where :math:`\alpha` is a scaling parameter and :math:`\beta` and
  :math:`\gamma` are shape parameters.


Matern Kernel
.............

.. class:: MaternKernel([ν::Real=1 [,θ::Real=1]]) <: MercerKernel

  The Matern kernel is a **Mercer** kernel [ras]_ given by:

  .. math::

      \kappa(\mathbf{x},\mathbf{y}) = \frac{1}{2^{\nu-1}\Gamma(\nu)} \left(\frac{2\sqrt{\nu}||\mathbf{x}-\mathbf{y}||}{\theta}\right)^{\nu} K_{\nu}\left(\frac{2\sqrt{\nu}||\mathbf{x}-\mathbf{y}||}{\theta}\right)

  where :math:`\Gamma` is the gamma function, :math:`K_{\nu}` is the modified 
  Bessel function of the second kind, :math:`\nu > 0` and :math:`\theta > 0`.


Linear Kernel
.............

.. class:: LinearKernel([a::Real=1 [,c::Real=1]]) <: MercerKernel

  The linear kernel is a **Mercer** kernel given by:

.. math::

    \kappa(\mathbf{x},\mathbf{y}) = a \mathbf{x}^\intercal \mathbf{y} + c \qquad \alpha > 0, \; c \geq 0


Polynomial Kernel
.................

.. class:: PolynomialKernel([a::Real=1 [,c::Real=1 [,d::Integer=3]]]) <: MercerKernel

  The polynomial kernel is a **Mercer** kernel given by:

  .. math::

    \kappa(\mathbf{x},\mathbf{y}) = (a \mathbf{x}^\intercal \mathbf{y} + c)^d \qquad \alpha > 0, \; c \geq 0, \; d \in \mathbb{Z}_{+}


Exponentiated Kernel
....................

.. class:: ExponentiatedKernel([a::Real=1]) <: MercerKernel

  The exponentiated kernel is a **Mercer** kernel given by:

  .. math::

    \kappa(\mathbf{x},\mathbf{y}) = \exp\left(a \mathbf{x}^\intercal \mathbf{y} \right) \qquad a > 0


Periodic Kernel
...............

.. class:: PeriodicKernel([α::Real=1 [,p::Real=π]]) <: MercerKernel

  The periodic kernel is given by:

  .. math::

      \kappa(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha \sum_{i=1}^n \sin(p(x_i - y_i))^2\right) \qquad p >0, \; \alpha > 0

  where :math:`\mathbf{x}` and :math:`\mathbf{y}` are :math:`n` dimensional 
  vectors. The parameters :math:`p` and :math:`\alpha` are scaling parameters 
  for the periodicity and the magnitude, respectively. This kernel is useful 
  when data has periodicity to it. 



Sigmoid Kernel
..............

.. class:: SigmoidKernel([a::Real=1 [,c::Real=1]]) <: Kernel

  The sigmoid kernel is given by:

  .. math::

      \kappa(\mathbf{x},\mathbf{y}) = \tanh(a \mathbf{x}^\intercal \mathbf{y} + c) \qquad \alpha > 0, \; c \geq 0

  The sigmoid kernel is a not a true kernel, although it has been used in 
  application. 
