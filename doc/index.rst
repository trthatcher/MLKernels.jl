========================
Machine Learning Kernels
========================

`MLKernels.jl`_ is a Julia package for Mercer kernel functions (or the 
covariance functions used in Gaussian processes) that are used in the kernel 
methods of machine learning. This package provides a flexible datatype for 
representing and constructing machine learning kernels as well as an efficient 
set of methods to compute or approximate kernel matrices. The package has no 
dependencies beyond base Julia.

Overview
========

The following lists the functions exported by `MLKernels.jl`_:

.. toctree::
    :maxdepth: 1

    interface

A number of kernels common to various domains have been pre-defined for ease of
use. A list of the pre-defined kernels is available below: 

.. toctree::
    :maxdepth: 1

    kernels

Kernel classes composed with the preceding kernels can be used to construct new
kernels. Affine combinations and point-wise products may also be used to
further construct kernels with various properties. See below for a list of the
pre-defined kernel classes as well as descriptions of the allowed kernel
operations:

.. toctree::
    :maxdepth: 1

    kernelclasses

Lastly, a basic introduction to the kernel methods is available on the page
below:

.. toctree::
    :maxdepth: 1

    theory

Kernels
=======

For quick reference, a list of default kernels is available below:

* :ref:`kern-scprod`
* :ref:`kern-sqdist`

Citations
=========

.. [berg] Berg C, Christensen JPR, Ressel P. 1984. *Harmonic Analysis on Semigroups*. Springer-Verlag New York. Chapter 3, General Results on Positive and Negative Definite Matrices and Kernels; p. 66-85.

.. [bou] Bouboulis P. 2014. *Academic Press Library in Signal Processing, Volume 1: Array and Statistical Signal Processing (1st ed.)*. Academic Press. Chapter 17, Online Learning in Reproducing Kernel Hilbert Spaces; p. 883-987.

.. [ram] Rasmussen C, Williams CKI. 2005. *Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning)*. The MIT Press. Chapter 4, Covariance Functions; p. 79-104.

.. _MLKernels.jl: https://github.com/trthatcher/MLKernels.jl
