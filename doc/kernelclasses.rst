--------------
Kernel Classes
--------------

Mercer Classes
--------------

Exponential Class
.................

.. math::

    k(\mathbf{x},\mathbf{y}) = \exp\left(-\alpha \kappa(\mathbf{x},\mathbf{y})^{\gamma}\right) \qquad \alpha > 0, \; 0 < \gamma \leq 1

Rational-Quadratic Class
........................

.. math::

    k(\mathbf{x},\mathbf{y}) = \left(1 +\alpha \kappa(\mathbf{x},\mathbf{y})^{\gamma}\right)^{-\beta} \qquad \alpha > 0, \; \beta > 0, \; 0 < \gamma \leq 1

Matern Class
.............

.. math::

    k(\mathbf{x},\mathbf{y}) = \frac{1}{2^{\nu-1}\Gamma(\nu)} \left(\frac{2\sqrt{\nu}\kappa(\mathbf{x},\mathbf{y})}{\theta}\right)^{\nu} K_{\nu}\left(\frac{2\sqrt{\nu}\kappa(\mathbf{x},\mathbf{y})}{\theta}\right)

Polynomial Class
................

The polynomial kernel is given by:

.. math::

    k(\mathbf{x},\mathbf{y}) = (\alpha\kappa(\mathbf{x},\mathbf{y}) + c)^d \qquad \alpha > 0, \; c \geq 0, \; d \in \mathbb{Z}_{+}


Exponentiated Class
...................

.. math::

    k(\mathbf{x},\mathbf{y}) = \exp(\alpha\kappa(\mathbf{x},\mathbf{y}) + c) \qquad \alpha > 0, \; c \geq 0

where :math:`\kappa` is a Mercer kernel. An exponentiated kernel is a Mercer kernel.


Negative Definite Classes
-------------------------

Power Kernel
............

.. math::

    k(\mathbf{x},\mathbf{y}) = \kappa(\mathbf{x},\mathbf{y})^{\gamma} \qquad 0 < \gamma \leq 1

where :math:`\kappa` is a non-negative negative definite kernel. The power kernel is a
negative definite kernel.

Log Kernel
..........

.. math::

    k(\mathbf{x},\mathbf{y}) = \log(1 + \alpha\kappa(\mathbf{x},\mathbf{y})^{\gamma}) \qquad \alpha > 0, \; 0 < \gamma \leq 1

where :math:`\kappa` is a non-negative negative definite kernel. The power kernel is a
negative definite kernel.


Other Classes
-------------

Sigmoid Kernel
..............

Construct a sigmoid kernel:

.. math::

    k(\mathbf{x},\mathbf{y}) = \tanh(\alpha\kappa(\mathbf{x},\mathbf{y}) + c) \qquad \alpha > 0, \; c \geq 0

where :math:`\kappa` is a Mercer kernel. The sigmoid kernel is a not a true kernel, although
it has been used in application.

-----------------
Kernel Operations
-----------------

Kernel Affinity
---------------

Kernel Sum
----------

Kernel Product
--------------
