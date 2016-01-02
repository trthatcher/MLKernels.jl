========================
Machine Learning Kernels
========================

`MLKernels.jl`_ is a Julia package for Mercer kernel functions (or the 
covariance functions of Gaussian processes) that are used in the kernel 
methods of machine learning. This package provides a flexible datatype for 
representing and constructing machine learning kernels as well as an efficient 
set of methods to compute or approximate kernel matrices. The package has no 
dependencies beyond base Julia.

Contents
========

.. toctree::
    :maxdepth: 2

    exportedfunctions
    kernelfunctions
    kernelclasses
    kernelmethods

Citations
=========

.. [berg] Berg C, Christensen JPR, Ressel P. 1984. *Harmonic Analysis on Semigroups*. Springer-Verlag New York. Chapter 3, General Results on Positive and Negative Definite Matrices and Kernels; p. 66-85.

.. [bou] Bouboulis P. 2014. *Academic Press Library in Signal Processing, Volume 1: Array and Statistical Signal Processing (1st ed.)*. Academic Press. Chapter 17, Online Learning in Reproducing Kernel Hilbert Spaces; p. 883-987.

.. [ras] Rasmussen C, Williams CKI. 2005. *Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning)*. The MIT Press. Chapter 4, Covariance Functions; p. 79-104.

.. _MLKernels.jl: https://github.com/trthatcher/MLKernels.jl
