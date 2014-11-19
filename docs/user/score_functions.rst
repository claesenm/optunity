Score and loss functions
========================

Score/loss functions are used to quantify the performance of a given model. Score functions are typically maximized (e.g. accuracy, concordance, ...) whereas
loss functions should be minimized (e.g. mean squared error, error rate, ...). Optunity provides common score/loss functions for your convenience.

For non-parameterized score/loss functions, we use the following calling convention: 

-   `y` (iterable): the true labels/function values
-   `yhat` (iterable): the predicted labels/function values
-   we assume `y` and `yhat` are of the same length (though we do not assert this).

For parameterized score/loss functions, we provide factories that return callables with the same signature as above.

All functions listed here are available in :mod:`optunity.score_functions`.

Score functions
---------------

Score functions are typically maximized (e.g. :func:`optunity.maximize`).

Classification
^^^^^^^^^^^^^^

+----------------------+---------------------------------------------+------+
| Score                | Associated Optunity function                | par. |
+======================+=============================================+======+
| accuracy             | :func:`~optunity.score_functions.accuracy`  | no   |
+----------------------+---------------------------------------------+------+
| area under ROC curve | :func:`~optunity.score_functions.auroc`     | yes  |
+----------------------+---------------------------------------------+------+
| area under PR curve  | :func:`~optunity.score_functions.aupr`      | yes  |
+----------------------+---------------------------------------------+------+
| Brier score          | :func:`~optunity.score_functions.brier`     | no   |
+----------------------+---------------------------------------------+------+
| :math:`F_\beta`      | :func:`~optunity.score_functions.fbeta`     | yes  |
+----------------------+---------------------------------------------+------+
| log loss             | :func:`~optunity.score_functions.logloss`   | no   |
+----------------------+---------------------------------------------+------+
| precision/PPV        | :func:`~optunity.score_functions.precision` | no   |
+----------------------+---------------------------------------------+------+
| recall/sensitivity   | :func:`~optunity.score_functions.recall`    | no   |
+----------------------+---------------------------------------------+------+
| specificity/NPV      | :func:`~optunity.score_functions.npv`       | no   |
+----------------------+---------------------------------------------+------+
| PU score             | :func:`~optunity.score_functions.pu_score`  | no   |
+----------------------+---------------------------------------------+------+

Regression
^^^^^^^^^^^

+----------------------+---------------------------------------------+------+
| Score                | Associated Optunity function                | par. |
+======================+=============================================+======+
| :math:`R^2`          | :func:`~optunity.score_functions.r2`        | no   |
+----------------------+---------------------------------------------+------+

Loss functions
---------------

Loss functions are typically minimized (e.g. :func:`optunity.minimize`).

Classification
^^^^^^^^^^^^^^^

+----------------------+---------------------------------------------+------+
| Score                | Associated Optunity function                | par. |
+======================+=============================================+======+
| error rate         | :func:`~optunity.score_functions.error_rate`  | no   |
+----------------------+---------------------------------------------+------+

Regression
^^^^^^^^^^^

+----------------------+---------------------------------------------+------+
| Score                | Associated Optunity function                | par. |
+======================+=============================================+======+
| mean squared error   | :func:`~optunity.score_functions.mse`       | no   |
+----------------------+---------------------------------------------+------+
