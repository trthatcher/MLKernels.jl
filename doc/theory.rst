.. _pagetheory:

Kernel Methods
==============

--------------
Kernel Methods
--------------

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

