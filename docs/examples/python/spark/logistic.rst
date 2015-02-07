Logistic regression with Spark and MLlib
============================================

.. include:: /global.rst

In this example, we will train a linear logistic regression model using Spark and MLlib. In this case, we have to tune one hyperparameter: `regParam` for L2 regularization.
We will use 5-fold cross-validation to find optimal hyperparameters.

In this example, we will use |maximize|, which by default uses Particle Swarms. Assumes the data is located in file `sample_svm_data.txt`, change it if necessary.

.. literalinclude:: /examples/python/spark/logistic.py
    :language: python
    :emphasize-lines: 19
