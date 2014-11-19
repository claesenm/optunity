Score and loss functions
========================

Score/loss functions are used to quantify the performance of a given model. Score functions are typically maximized (e.g. accuracy, concordance, ...) whereas
loss functions should be minimized (e.g. mean squared error, error rate, ...). Optunity provides common score/loss functions for your convenience.

We use the following calling convention: 

-   `y` (iterable): the true labels/function values
-   `yhat` (iterable): the predicted labels/function values
-   we assume `y` and `yhat` are of the same length (though we do not assert this).
-   potential parameters of the score function must be added by keyword

All functions listed here are available in :mod:`optunity.score_functions`.

Score functions
---------------

Score functions are typically maximized (e.g. :func:`optunity.maximize`).

Classification
^^^^^^^^^^^^^^

+----------------------+---------------------------------------------+
| Score                | Associated Optunity function                |
+======================+=============================================+
| accuracy             | :func:`~optunity.score_functions.accuracy`  |
+----------------------+---------------------------------------------+
| area under ROC curve | :func:`~optunity.score_functions.roc_auc`   |
+----------------------+---------------------------------------------+
| area under PR curve  | :func:`~optunity.score_functions.pr_auc`    |
+----------------------+---------------------------------------------+
| convex hull of ROC   | :func:`~optunity.score_functions.roc_hull`  |
+----------------------+---------------------------------------------+
| :math:`F_\beta`      | :func:`~optunity.score_functions.fbeta`     |
+----------------------+---------------------------------------------+
| precision/PPV        | :func:`~optunity.score_functions.precision` |
+----------------------+---------------------------------------------+
| recall/sensitivity   | :func:`~optunity.score_functions.recall`    |
+----------------------+---------------------------------------------+
| specificity/NPV      | :func:`~optunity.score_functions.npv`       |
+----------------------+---------------------------------------------+
| PU score             | :func:`~optunity.score_functions.pu_score`  |
+----------------------+---------------------------------------------+

Regression
^^^^^^^^^^^

+----------------------+---------------------------------------------+
| Score                | Associated Optunity function                |
+======================+=============================================+
| :math:`R^2`          | :func:`~optunity.score_functions.r2`        |
+----------------------+---------------------------------------------+

Loss functions
---------------

Loss functions are typically minimized (e.g. :func:`optunity.minimize`).

Classification
^^^^^^^^^^^^^^^

+----------------------+----------------------------------------------+
| Score                | Associated Optunity function                 |
+======================+==============================================+
| Brier score          | :func:`~optunity.score_functions.brier`      |
+----------------------+----------------------------------------------+
| error rate           | :func:`~optunity.score_functions.error_rate` |
+----------------------+----------------------------------------------+
| log loss             | :func:`~optunity.score_functions.logloss`    |
+----------------------+----------------------------------------------+

Regression
^^^^^^^^^^^

+----------------------+---------------------------------------------+
| Score                | Associated Optunity function                |
+======================+=============================================+
| mean squared error   | :func:`~optunity.score_functions.mse`       |
+----------------------+---------------------------------------------+
