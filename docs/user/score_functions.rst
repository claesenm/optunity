Score and loss functions
========================

Score/loss functions are used to quantify the performance of a given model. Score functions are typically maximized (e.g. accuracy, concordance, ...) whereas
loss functions should be minimized (e.g. mean squared error, error rate, ...). Optunity provides common score/loss functions for your convenience.

For non-parameterized score/loss functions, we use the following calling convention: 

-   `y` (iterable): the true labels/function values
-   `yhat` (iterable): the predicted labels/function values

We assume `y` and `yhat` are of the same length (though we do not assert this).

For parameterized score/loss functions, we provide factories that return callables with the same signature as above.

The following functions are available in :mod:`optunity.score_functions`:

Score functions
---------------

*   Classification

    -   accuracy
    -   area under the ROC curve
    -   area under the PR curve
    -   log loss
    -   Brier score
    -   :math:`F_\beta`
    -   PU score
    -   precision/positive predictive value
    -   recall/sensitivity
    -   specificity/negative predictive value

*   Regression

    -   :math:`R^2`

Loss functions
---------------

*   Classification

    -   error rate

*   Regression

    -   mean squared error

