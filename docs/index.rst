.. image:: /logo.png
    :alt: Optunity

.. image:: https://travis-ci.org/claesenm/optunity.svg?branch=master
    :target: https://travis-ci.org/claesenm/optunity
    :align: right

==========

.. include:: /global.rst

Optunity is a library containing various optimizers for hyperparameter tuning.
Hyperparameter tuning is a recurrent problem in many machine learning tasks,
both supervised and unsupervised.This package provides several distinct approaches 
to solve such problems including some helpful facilities such as cross-validation 
and a plethora of score functions.

.. sidebar:: **Getting started**

    * :doc:`Installation <user/installation>`
    * :doc:`User guide <user/index>`
    * `Report a problem <https://github.com/claesenm/optunity/issues/new>`_


    **Obtaining Optunity**

    * at |pypi| (releases)
    * at |github| (development)

From an optimization point of view, the tuning problem can be considered as 
follows: the objective function is non-convex, non-smooth and 
typically expensive to evaluate. Tuning examples include optimizing regularization 
or kernel parameters.

The Optunity library is implemented in Python and allows straightforward integration in other machine learning environments. 
Optunity is currently also supported in |wrapper-r|, |wrapper-matlab| and Java through Jython.

Optunity is free software, using a BSD-style license.

Example
-----------

As a simple example of Optunity's features, the code below demonstrates how to tune an SVM with RBF kernel using Optunity and scikit-learn. 
This involves optimizing the hyperparameters ``gamma`` and ``C``:

.. literalinclude:: /examples/python/sklearn/svc.py
    :language: python
    :emphasize-lines: 6,13

For more examples, please see our |examples|.

Quick setup
----------------

Issue the following commands to get started on Linux:

.. code-block:: bash

    git clone https://github.com/claesenm/optunity.git
    export PYTHONPATH=$PYTHONPATH:$(pwd)/optunity/

Afterwards, importing ``optunity`` should work in Python:

.. code-block:: bash

    python -c 'import optunity'

For a proper installation, run the following:

.. code-block:: bash

    python optunity/setup.py install

or, if you have pip:

.. code-block:: bash

    pip install optunity

Installation may require superuser priviliges.

Developer Guide
-----------------

|api|


Contributors
------------

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
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
    :maxdepth: 2
    :hidden:

    /user/installation
    /user/index
    /examples/index
    /wrappers/index
    /api/optunity
