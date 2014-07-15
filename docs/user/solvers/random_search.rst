===========
Optunity
===========

..
    .. toctree::
    :maxdepth: 4

Optunity is a library containing various optimizers for hyperparameter tuning.
Hyperparameter tuning is a recurrent problem in many machine learning tasks,
both supervised and unsupervised.This package provides several distinct approaches 
to solve such problems including some helpful facilities such as cross-validation 
and a plethora of score functions.

From an optimization point of view, the tuning problem can be considered as 
follows: the objective function is non-convex, non-differentiable and 
typically expensive to evaluate. Tuning examples include optimizing regularization 
or kernel parameters.

The Optunity library is implemented in Python and allows straightforward
integration in other machine learning environments, including R and MATLAB.

.. sidebar:: Quick setup

    Issue the following commands to get started on Linux::

        git clone https://github.com/claesenm/optunity.git
        export PYTHONPATH=#PYTHONPATH:$(pwd)/optunity/

    Afterwards, importing ``optunity`` should work in Python::

        python -c 'import optunity'

    For a proper installation, run the following::

        python optunity/setup.py install

    Installation may require superuser priviliges.

Optunity is free software, using a BSD license.

User Guide
==============

**First steps**

* :doc:`Overview (Start Here!) <user/overview>`
* :doc:`Installation <user/installation>`

-------------

**Examples of Optunity usage**

* :doc:`Optimizing a simple function <user/examples/parabola2d>`
* :doc:`Tuning SVM hyperparameters <user/examples/svm>`

For more examples, please see our :doc:`examples page <user/examples/index>`.

-------------

**Using Optunity in different environments**

* :doc:`MATLAB <wrappers/matlab/index>`
* :doc:`R <wrappers/R/index>`


Developer Guide
=================

- :doc:`Developer guide <dev/index>`
- :doc:`API reference <optunity>`

Contributors
============

Optunity is developed at the STADIUS lab of the dept. of electrical engineering at 
KU Leuven (ESAT). 
The main contributors to Optunity are:

**Marc Claesen**

    - Python package
    - framework design & implementation
    - solver implementation
    - communication protocol design & implementation
    - MATLAB wrapper

**Jaak Simm**

    - communication protocol design
    - R wrapper

**Dusan Popovic**

    - code examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
  :hidden:
