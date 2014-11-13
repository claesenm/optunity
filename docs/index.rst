.. image:: /logo.png
    :alt: Optunity

.. image:: https://travis-ci.org/claesenm/optunity.svg?branch=master
    :target: https://travis-ci.org/claesenm/optunity
    :align: right

==========

Optunity is a library containing various optimizers for hyperparameter tuning.
Hyperparameter tuning is a recurrent problem in many machine learning tasks,
both supervised and unsupervised.This package provides several distinct approaches 
to solve such problems including some helpful facilities such as cross-validation 
and a plethora of score functions.

.. sidebar:: **Getting started**

    * :doc:`Overview <user/index>`
    * :doc:`Installation <user/installation>`
    * `Report a problem <https://github.com/claesenm/optunity/issues/new>`_

    **Obtaining Optunity**

    * at PyPI_ (releases)
    * at GitHub_ (development)

.. _PyPI: https://pypi.python.org/pypi/Optunity
.. _GitHub: https://github.com/claesenm/optunity

From an optimization point of view, the tuning problem can be considered as 
follows: the objective function is non-convex, non-smooth and 
typically expensive to evaluate. Tuning examples include optimizing regularization 
or kernel parameters.

The Optunity library is implemented in Python and allows straightforward
integration in other machine learning environments. Optunity will soon become
available in :doc:`R </wrappers/R/index>` and :doc:`MATLAB </wrappers/matlab/index>`.

Optunity is free software, using a BSD-style license.

User Guide
==============

**Examples of Optunity usage**

Tuning an SVM with RBF kernel using Optunity and scikit-learn:::

    import optunity
    import sklearn.svm

    @optunity.cross_validated(x=data, y=labels, num_folds=10, num_iter=2)
    def svm acc(x_train, y_train, x_test, y_test, C, gamma):
        model = sklearn.svm.SVC(C=C, gamma=gamma).fit(x_train, y_train)
        y_pred = model.predict(x_test)
        return optunity.score_functions.accuracy(y_test, y_pred)

    optimal_pars, _, _ = optunity.maximize(svm acc, num_evals=200, C=[0, 10], gamma=[0, 1])
    optimal_model = sklearn.svm.SVC(**optimal_pars).fit(data, labels)

For more examples, please see our :doc:`examples page </examples/index>`.

-------------

**Quick setup**

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
=================

- :doc:`Developer guide <dev/index>`
- :doc:`API reference <api/optunity>`

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
    :maxdepth: 2
    :includehidden:

    /user/installation
    /user/index
    /examples/index
    /wrappers/index
    /dev/index
    /api/optunity
