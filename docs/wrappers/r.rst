R
=======

.. include:: /global.rst
.. highlight:: r

In this page we briefly discuss the R wrapper, which provides most of Optunity's functionality. 
For a general overview, we recommend reading the :doc:`/user/index`.

For installation instructions, please refer to |installation|. To use the package has to be loaded like all R packages by::

    library(optunity)


Manual
--------

All functions in R wrapper have documentation available in R's help system,
e.g., `?cv.particle_swarm` gives help for cross-validation (CV) using **particle swarms**.
To see the list of available functions type `optunity::<TAB><TAB>` in R's command line.

For R following main functions are available:

-   `cv.setup` for **setting up CV**, specifying data and the number of folds

-   `cv.particle_swarm`, `cv.nelder_mead`, `cv.grid_search`, `cv.random_search` for **optimizing hyperparameters** in CV setting.

-   `auc_roc` and `auc_pr` for calculating **AUC** of ROC and precision-recall curve.

-   `early_rie` and `early_bedroc` for **early discovery** metrics.

-   `mean_se` and `mean_ae` for regression **loss functions**, mean squared error and mean absolute error.

-   `particle_swarm`, `nelder_mead`, `grid_search`, `random_search` for minimizing (or maximizing) regular functionals.

-   `cv.run` performs a **cross-validation** evaluation for the given values of the hyperparameters (no optimization).

General workflow is to first create a CV object using **cv.setup** and then use
it to run CV, by using functions like **cv.particle_swarms**, **cv.random_search** etc.

Examples
----------
Please see the example pages,
:doc:`/examples/r/ridgereg` and :doc:`/examples/r/svm`. The R help pages for each function
contain examples showing how to use them.

