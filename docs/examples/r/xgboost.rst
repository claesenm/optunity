xgboost
============================================

.. include:: /global.rst
.. highlight:: r

In this example, we will train a xgboost. There is only one hyper-parameter `max.depth`, which takes integer values.
Therefore, we will use grid search to find `max.depth` that maximizes AUC-ROC in twice iterated 5-fold cross-validation::

    library(optunity)
    library(xgboost)

    data(agaricus.train, package='xgboost')
    data(agaricus.test, package='xgboost')
    train <- agaricus.train
    test  <- agaricus.test

    learn_xg <- function(x, y, xtest, ytest, max.depth) {
      ## train xgboost:
      bst <- xgboost(data = x, label = y, max.depth = round(max.depth),
                   eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
      predict(bst, xtest)
    }

    ## using grid search to find max.depth maximizing AUC-ROC
    ## grid for max.depth is 2:10
    cv <- cv.setup(x = train$data, y = train$label, score=auc_roc, num_folds = 5, num_iter = 2)

    ## running cross-validation
    res.grid <- cv.grid_search(cv, learn_xg, max.depth = 2:10, maximize=TRUE)

    ## best result:
    res.grid$solution$max.depth

    ## train xgboost with the best max.depth
    xbest <- xgboost(data = train$data, label = train$label, 
                     max.depth = res.grid$solution$max.depth,
                     eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")

    ## check AUC-ROC of the best model
    pred <- predict(xbest, test$data)
    auc_roc(test$label, pred)


Here we used grid of 2:10 for `max.depth`.
