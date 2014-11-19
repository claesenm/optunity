Score and loss functions
========================

Score/loss functions are used to quantify the performance of a given model. Score functions are typically maximized (e.g. accuracy, concordance, ...) whereas
loss functions should be minimized (e.g. mean squared error, error rate, ...). Optunity provides common score/loss functions for your convenience.

The following functions are available in :mod:`optunity.score_functions`:
-   Score functions

    *   Classification

        -   accuracy
        -   area under the ROC curve
        -   area under the PR curve
        -   log loss
        -   Brier score
        -   F:math:`\alpha`

    *   Regression

-   Loss functions

    *   Classification

        -   error rate

    *   Regression

        -   mean squared error

Score functions
----------------


Loss functions
---------------
