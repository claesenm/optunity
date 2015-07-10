
sklearn: automated learning method selection and tuning
=======================================================

In this tutorial we will show how to use Optunity in combination with
sklearn to classify the digit recognition data set available in sklearn.
The cool part is that we will use Optunity to choose the best approach
from a set of available learning algorithms and optimize hyperparameters
in one go. We will use the following learning algorithms:

-  k-nearest neighbour

-  SVM

-  Naive Bayes

-  Random Forest

For simplicity, we will focus on a binary classification task, namely
digit 3 versus digit 9. We start with the necessary imports and create
the data set.

.. code:: python

    import optunity
    import optunity.metrics
    import numpy as np
    
    # k nearest neighbours
    from sklearn.neighbors import KNeighborsClassifier
    # support vector machine classifier
    from sklearn.svm import SVC 
    # Naive Bayes
    from sklearn.naive_bayes import GaussianNB 
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier 
    
    from sklearn.datasets import load_digits
    digits = load_digits()
    n = digits.data.shape[0]
    
    positive_digit = 3
    negative_digit = 9
    
    positive_idx = [i for i in range(n) if digits.target[i] == positive_digit]
    negative_idx = [i for i in range(n) if digits.target[i] == negative_digit]
    
    # add some noise to the data to make it a little challenging
    original_data = digits.data[positive_idx + negative_idx, ...]
    data = original_data + 5 * np.random.randn(original_data.shape[0], original_data.shape[1])
    labels = [True] * len(positive_idx) + [False] * len(negative_idx)
For the SVM model we will let Optunity optimize the kernel family,
choosing from linear, polynomial and RBF. We start by creating a
convenience functions for SVM training that handles this:

.. code:: python

    def train_svm(data, labels, kernel, C, gamma, degree, coef0):
        """A generic SVM training function, with arguments based on the chosen kernel."""
        if kernel == 'linear':
            model = SVC(kernel=kernel, C=C)
        elif kernel == 'poly':
            model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
        elif kernel == 'rbf':
            model = SVC(kernel=kernel, C=C, gamma=gamma)
        else: 
            raise ArgumentError("Unknown kernel function: %s" % kernel)
        model.fit(data, labels)
        return model
Every learning algorithm has its own hyperparameters:

-  k-NN: :math:`1 < n\_neighbors < 5` the number of neighbours to use

-  SVM: kernel family and misclassification penalty, we will make the
   penalty contingent on the family. Per kernel family, we have
   different hyperparameters:

   -  polynomial kernel: :math:`0 < C < 50`, :math:`2 < degree < 5` and
      :math:`0 < coef0 < 1`
   -  linear kernel: :math:`0 < C < 2`,
   -  RBF kernel: :math:`0 < C < 10` and :math:`0 < gamma < 1`

-  naive Bayes: no hyperparameters

-  random forest:

   -  :math:`10 < n\_estimators < 30`: number of trees in the forest
   -  :math:`5 < max\_features < 20`: number of features to consider for
      each split

This translates into the following search space:

.. code:: python

    search = {'algorithm': {'k-nn': {'n_neighbors': [1, 5]},
                            'SVM': {'kernel': {'linear': {'C': [0, 2]},
                                               'rbf': {'gamma': [0, 1], 'C': [0, 10]},
                                               'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1]}
                                               }
                                    },
                            'naive-bayes': None,
                            'random-forest': {'n_estimators': [10, 30],
                                              'max_features': [5, 20]}
                            }
             }
We also need an objective function that can properly orchestrate
everything. We will choose the best model based on area under the ROC
curve in 5-fold cross-validation.

.. code:: python

    @optunity.cross_validated(x=data, y=labels, num_folds=5)
    def performance(x_train, y_train, x_test, y_test,
                    algorithm, n_neighbors=None, n_estimators=None, max_features=None,
                    kernel=None, C=None, gamma=None, degree=None, coef0=None):
        # fit the model
        if algorithm == 'k-nn':
            model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
            model.fit(x_train, y_train)
        elif algorithm == 'SVM':
            model = train_svm(x_train, y_train, kernel, C, gamma, degree, coef0)
        elif algorithm == 'naive-bayes':
            model = GaussianNB()
            model.fit(x_train, y_train)
        elif algorithm == 'random-forest':
            model = RandomForestClassifier(n_estimators=int(n_estimators),
                                           max_features=int(max_features))
            model.fit(x_train, y_train)
        else:
            raise ArgumentError('Unknown algorithm: %s' % algorithm)
    
        # predict the test set
        if algorithm == 'SVM':
            predictions = model.decision_function(x_test)
        else:
            predictions = model.predict_proba(x_test)[:, 1]
    
        return optunity.metrics.roc_auc(y_test, predictions, positive=True)
Lets do a simple test run of this fancy objective function.

.. code:: python

    performance(algorithm='k-nn', n_neighbors=3)



.. parsed-literal::

    0.9547920006472639



Seems okay! Now we can let Optunity do its magic with a budget of 300
tries.

.. code:: python

    optimal_configuration, info, _ = optunity.maximize_structured(performance, 
                                                                  search_space=search, 
                                                                  num_evals=300)
    print(optimal_configuration)
    print(info.optimum)

.. parsed-literal::

    {'kernel': 'poly', 'C': 38.5498046875, 'algorithm': 'SVM', 'degree': 3.88525390625, 'n_neighbors': None, 'n_estimators': None, 'max_features': None, 'coef0': 0.71826171875, 'gamma': None}
    0.979302949566


Finally, lets make the results a little bit more readable. All
dictionary items in ``optimal_configuration`` with value ``None`` can be
removed.

.. code:: python

    solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
    print('Solution\n========')
    print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))

.. parsed-literal::

    Solution
    ========
    kernel 	 poly
    C 	 38.5498046875
    coef0 	 0.71826171875
    degree 	 3.88525390625
    algorithm 	 SVM

