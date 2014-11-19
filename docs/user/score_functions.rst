Score and loss functions
========================

Score/loss functions are used to quantify the performance of a given model. Score functions are typically maximized (e.g. accuracy, concordance, ...) whereas
loss functions should be minimized (e.g. mean squared error, error rate, ...). Optunity provides common score/loss functions for your convenience.

For non-parameterized score/loss functions, we use the following calling convention: 

-   `y` (iterable): the true labels/function values
-   `yhat` (iterable): the predicted labels/function values
-   we assume `y` and `yhat` are of the same length (though we do not assert this).

For parameterized score/loss functions, we provide factories that return callables with the same signature as above.

The following functions are available in :mod:`optunity.score_functions`:

Score functions
---------------

Score functions are typically maximized (e.g. :func:`optunity.maximize`).

Classification
^^^^^^^^^^^^^^

+----------------------+--------------------------------------------------------+----------------+
| Score                | Associated Optunity function                           | Parameterized? |
+======================+========================================================+================+
| accuracy             | :func:`accuracy <optunity.score_functions.accuracy>`   | no  |
| area under ROC curve | :func:`auroc <optunity.score_functions.auroc>`         | no  |
| area under PR curve  | :func:`aupr <optunity.score_functions.aupr>`           | no  |
| Brier score          | :func:`brier <optunity.score_functions.brief>`         | no  |
| :math:`F_\beta`      | :func:`fbeta <optunity.score_functions.fbeta>`         | yes |
| log loss             | :func:`logloss <optunity.score_functions.logloss>`     | no  |
| precision/PPV        | :func:`precision <optunity.score_functions.precision>` | no  |
| recall/sensitivity   | :func:`recall <optunity.score_functions.recall>`       | no  |
| specificity/NPV      | :func:`npv <optunity.score_functions.npv>`             | no  |
+----------------------+--------------------------------------------------------+----------------+

Regression
^^^^^^^^^^^

-   :math:`R^2`

Loss functions
---------------

Loss functions are typically minimized (e.g. :func:`optunity.minimize`).

Classification
^^^^^^^^^^^^^^^

-   error rate

Regression
^^^^^^^^^^^

-   mean squared error

