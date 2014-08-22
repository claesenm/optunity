
context("Cross-validation")

x <- matrix(runif(50*5), 50, 5)
y <- x[,1] + 0.5*x[,2] + 0.1*runif(50)

test_that("cv.setup can be created", {
  cv <- cv.setup(x, y, score=score.neg.mse, num_folds = 10, num_iter = 2)
  expect_equal( cv$supervised, TRUE )
  expect_equal( nrow(cv$folds), 50 )
  expect_equal( ncol(cv$folds), 2 )
})

test_that("cv.run works with regression", {
  cv <- cv.setup(x, y, score=score.neg.mse, num_folds = 5, num_iter = 2)
  ## linear regression
  regr <- function(x, y, xtest, ytest) {
    beta = solve(t(x) %*% x, t(x) %*% y)
    xtest %*% beta
  }
  result <- cv.run(cv, regr)
})
