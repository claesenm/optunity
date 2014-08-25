
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
  expect_equal(nrow(result$scores), cv$num_folds)
  expect_equal(ncol(result$scores), cv$num_iter)
  expect_equal(length(result$score.iter.mean), cv$num_iter)
})

test_that("cv.run works with regression that has 1 param", {
  cv <- cv.setup(x, y, score=score.neg.mse, num_folds = 5, num_iter = 2)
  ## linear regression + L2 regularization
  regr <- function(x, y, xtest, ytest, reg) {
    beta = solve(t(x) %*% x + diag(x=reg, ncol(x)), t(x) %*% y)
    xtest %*% beta
  }
  result <- cv.run(cv, regr, reg = 2.5)
  expect_equal(nrow(result$scores), cv$num_folds)
  expect_equal(ncol(result$scores), cv$num_iter)
  expect_equal(length(result$score.iter.mean), cv$num_iter)
})

test_that("cv.run and grid_search over lambda",{
  ## linear regression + L2 regularization
  regr <- function(x, y, xtest, ytest, reg) {
    beta = solve(t(x) %*% x + diag(x=reg, ncol(x)), t(x) %*% y)
    xtest %*% beta
  }
  cv <- cv.setup(x, y, score=score.neg.mse, num_folds = 3, num_iter = 2)
  regr.cv <- function(reg) cv.run(cv, regr, reg = reg)$score.mean
  
  ## optimize
  result <- grid_search(regr.cv, reg=c(0, 1e-2, 1e-1))
  expect_equal(length(result$call_log$values), 3)
  expect_equal(result$stats$num_evals, 3)
})

test_that("cv.grid_search fails with wrong parameter", {
  cv <- cv.setup(x, y, score=score.neg.mse, num_folds = 3, num_iter = 2)
  regr <- function(x, y, xtest, ytest, reg) {
    beta = solve(t(x) %*% x + diag(x=reg, ncol(x)), t(x) %*% y)
    xtest %*% beta
  }
  expect_error( {cv.grid_search(cv, regr, noparam = c(1,2) )} )
})


test_that("cv.grid_search works", {
  regr <- function(x, y, xtest, ytest, reg) {
    beta = solve(t(x) %*% x + diag(x=reg, ncol(x)), t(x) %*% y)
    xtest %*% beta
  }
  cv <- cv.setup(x, y, score=score.neg.mse, num_folds = 3, num_iter = 2)
  res <- cv.grid_search(cv, regr, reg = c(0, 1e-2, 1e-1) )
})

d2 <- function(x1, x2) {
  nsq1 = rowSums(x1^2)
  nsq2 = rowSums(x2^2)
  matrix(nsq1, nrow(x1), nrow(x2)) + matrix(nsq2, nrow(x1), nrow(x2), byrow=T) - 2 * tcrossprod(x1, x2)
}

test_that("cv.grid_search works with 2 params", {
  regr <- function(x, y, xtest, ytest, reg, unused) {
    beta = solve(t(x) %*% x + diag(x=reg, ncol(x)), t(x) %*% y)
    xtest %*% beta
  }
  cv <- cv.setup(x, y, score=score.neg.mse, num_folds = 3, num_iter = 2)
  res <- cv.grid_search(cv, regr, reg = c(0, 1e-2, 1e-1), unused=c(1) )
})

