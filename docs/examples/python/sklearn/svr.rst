Support vector machine regression (SVR)
========================================

.. include:: /global.rst

You can find an executable version of this example in `bin/examples/python/sklearn/svc.py` in your Optunity release.

In this example, we will train an SVC with RBF kernel using scikit-learn. In this case, we have to tune two hyperparameters: `C` and `gamma`.
We will use twice iterated 10-fold cross-validation to test a pair of hyperparameters.

In this example, we will use |maximize|.

.. literalinclude:: /examples/python/sklearn/svr.py
    :language: python
    :emphasize-lines: 6,15
