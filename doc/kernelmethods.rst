----------------
The Kernel Trick
----------------

Many machine and statistical learning algorithms, such as support vector 
machines and principal components analysis, are based on **inner products**. These 
methods can often be generalized through use of the **kernel trick** to create a
nonlinear decision boundary without using an explicit mapping to another space. 

The kernel trick makes use of **Mercer kernels** which operate on vectors in the
input space but can be expressed as inner products in another space. In other
words, if :math:`\mathcal{X}` is the input vector space and :math:`\kappa` is
the Mercer kernel function, then for some vector space :math:`\mathcal{V}` there
exists a function :math:`\phi` such that:

.. math::

    \kappa(x_1, x_2) 
    = \left\langle \phi(x_1), \phi(x_2)\right\rangle_{\mathcal{V}}
    \qquad x_1, x_2 \in \mathcal{X}

In machine learning, the vector space :math:`\mathcal{X}` is known as the
feature space and the function :math:`\phi` is known as a feature map. A simple 
example of a feature map can be shown with the Polynomial Kernel:

.. math::

    \kappa(\mathbf{x},\mathbf{y}) = (a\mathbf{x}^\intercal\mathbf{y} + c)^{d}
    \qquad \mathbf{x}, \mathbf{y} \in \mathbb{R}^n, 
    \quad a, c \in \mathbb{R}_+
    \quad d \in \mathbb{Z}_+

In our example, we will use :math:`n=1`, :math:`d=2`, :math:`a=1` and
:math:`c=1/2`. Substituting these values in, we get the following kernel
function:

.. math::

    \kappa(x,y) = \left(xy + \frac{1}{2}\right)^2 = x^2y^2 + xy + \frac{1}{2^2}
    = \phi(x)^\intercal\phi(y)

Where the feature map :math:`\phi : \mathbb{R} \rightarrow \mathbb{R}^3` is
defined by:

.. math::

    \phi(x) = 
    \begin{bmatrix}
        x^2 \\
        x \\
        1/2
    \end{bmatrix}

The advantage of the implicit feature map is that we may transform non-linearly
data into linearly separable data in the implicit space. For example, suppose 
we have a single feature and two classes that cannot be separated using a 
linear function of that feature:

.. image:: images/kerneltrick/Feature.png

Using the feature map above, we create a data set that is linearly separable:

.. image:: images/kerneltrick/FeatureMap.png

-------
Kernels
-------

The kernel methods are a class of algorithms that are used for pattern analysis. These methods make
use of **kernel** functions. A symmetric, real valued kernel function 
:math:`\kappa: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}` is said to be **positive 
definite** or **Mercer** if and only:

.. math::

    \sum_{i=1}^n \sum_{j=1}^n c_i c_j \kappa(\mathbf{x}_i,\mathbf{x}_j) \geq 0

for all :math:`n \in \mathbb{N}`, :math:`\{\mathbf{x}_1, \dots, \mathbf{x}_n\} \subseteq \mathcal{X}`
and :math:`\{c_1, \dots, c_n\} \subseteq \mathbb{R}`. Similarly, a real valued kernel function
is said to be **negative definite** if and only if:

.. math::

    \sum_{i=1}^n \sum_{j=1}^n c_i c_j \kappa(\mathbf{x}_i,\mathbf{x}_j) \leq 0 \qquad \sum_{i=1}^n c_i = 0

for :math:`n \geq 2`, :math:`\{\mathbf{x}_1, \dots, \mathbf{x}_n\} \subseteq \mathcal{X}` and 
:math:`\{c_1, \dots, c_n\} \subseteq \mathbb{R}`. In machine learning literature, **conditionally
positive definite** kernels are often studied instead. This is simply a reversal of the above
inequality. Trivially, every negative definite kernel can be transformed into a conditionally
positive definite kernel by negation.


--------------
Nystrom method
--------------
