library("optunity")

## fold generation
strata <- list(c(1,2,3), c(6,7,8,9))
folds <- generate_folds(20, num_folds=10, num_iter=2, strata=strata)


## solver manual request
manual()

## solver demo
f <- function(x, y) return (-x^2-y^2)
cfg <- list(x = seq(-5, 5, by=0.5), y = seq(-5, 5, by=0.5))
result <- tune('grid search', cfg, f, return_call_log=TRUE)
