.. image:: logo/logo.png
    :alt: Optunity
    :align: left

.. image:: https://travis-ci.org/claesenm/optunity.svg?branch=master
    :target: https://travis-ci.org/claesenm/optunity
    :align: right

=========

Optunity is a library containing various optimizers for hyperparameter tuning.
Hyperparameter tuning is a recurrent problem in many machine learning tasks,
both supervised and unsupervised. Tuning examples include optimizing 
regularization or kernel parameters.

From an optimization point of view, the tuning problem can be considered as 
follows: the objective function is non-convex, non-differentiable and 
typically expensive to evaluate.

This package provides several distinct approaches to solve such problems including 
some helpful facilities such as cross-validation and a plethora of score functions.

The Optunity library is implemented in Python and allows straightforward
integration in other machine learning environments, including R and MATLAB.

To get started with Optunity on Linux, issue the following commands::

    git clone https://github.com/claesenm/optunity.git
    echo "export PYTHONPATH=$PYTHONPATH:$(pwd)/optunity" >> ~/.bashrc

Afterwards, importing ``optunity`` should work in Python::

    #!/usr/bin/env python
    import optunity

Optunity is developed at the STADIUS lab of the dept. of electrical engineering
at KU Leuven (ESAT). Optunity is free software, using a BSD license.

For more information, please refer to the following pages:
http://www.optunity.net

Contributors
============

The main contributors to Optunity are:

* Marc Claesen: framework design & implementation, communication infrastructure,
  MATLAB wrapper and all solvers.

* Jaak Simm: R wrapper.
