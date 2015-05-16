Sobol sequences
===============

.. include:: /global.rst

A Sobol sequence is a low discrepancy quasi-random sequence. Sobol sequences were designed to cover the unit 
hypercube with lower discrepancy than completely random sampling (e.g. |randomsearch|). Optunity supports Sobol
sequences in up to 40 dimensions (e.g. 40 hyperparameters). 

The figures below show the differences between a Sobol sequence and sampling uniformly at random.

.. figure:: sobol.png
    :alt: 100 points sampled in 2D with a Sobol sequence.

    100 points sampled in 2D with a Sobol sequence.



.. figure:: random.png
    :alt: 100 points sampled in 2D uniformly at random.

    100 points sampled in 2D uniformly at random.

The mathematical details on Sobol sequences are available in the following papers: [SOBOL]_, [ANTONOV]_, [BRATLEY]_, [FOX]_.
