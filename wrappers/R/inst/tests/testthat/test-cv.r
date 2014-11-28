
context("Cross-validation")

x <- matrix(runif(50*5), 50, 5)
y <- x[,1] + 0.5*x[,2] + 0.1*runif(50)

## linear regression
regr <- function(x, y, xtest, ytest, reg=0) {
  C =  diag(x=reg, ncol(x))
  beta = solve(t(x) %*% x + C, t(x) %*% y)
  xtest %*% beta
}

## linear regression with own scoring function
regr.score <- function(x, y, xtest, ytest, reg=0) {
  C =  diag(x=reg, ncol(x))
  beta = solve(t(x) %*% x + C, t(x) %*% y)
  sum((xtest %*% beta - ytest)^2)
}

## linear regression with 2 scoring functions
regr.score2 <- function(x, y, xtest, ytest, reg=0) {
  C =  diag(x=reg, ncol(x))
  beta = solve(t(x) %*% x + C, t(x) %*% y)
  yhat = xtest %*% beta
  return( c(
    sqerror  = sum((yhat - ytest)^2), 
    abserror = sum(abs(yhat - ytest))
  ))
}


test_that("cv.setup can be created", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 10, num_iter = 2)
  expect_equal( cv$supervised, TRUE )
  expect_equal( nrow(cv$folds), 50 )
  expect_equal( ncol(cv$folds), 2 )
})

test_that("cv.run works with regression", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 5, num_iter = 2)
  result <- cv.run(cv, regr)
  expect_equal(ncol(result$scores), cv$num_folds)
  expect_equal(nrow(result$scores), cv$num_iter)
  expect_equal(length(result$score.iter.mean), cv$num_iter)
})

test_that("1-iter cv.run works with regression", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 5, num_iter = 1)
  result <- cv.run(cv, regr)
  expect_equal(ncol(result$scores), cv$num_folds)
  expect_equal(nrow(result$scores), cv$num_iter)
  expect_equal(length(result$score.iter.mean), cv$num_iter)
})

test_that("cv.run works with regression that has 1 param", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 5, num_iter = 2)
  ## linear regression + L2 regularization
  result <- cv.run(cv, regr, reg = 2.5)
  expect_equal(ncol(result$scores), cv$num_folds)
  expect_equal(nrow(result$scores), cv$num_iter)
  expect_equal(length(result$score.iter.mean), cv$num_iter)
})

test_that("cv.run and grid_search over lambda",{
  ## linear regression + L2 regularization
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  regr.cv <- function(reg) cv.run(cv, regr, reg = reg)$score.mean
  
  ## optimize
  result <- grid_search(regr.cv, reg=c(0, 1e-2, 1e-1), maximize=FALSE)
  expect_equal(length(result$call_log$values), 3)
  expect_equal(result$stats$num_evals, 3)
})

test_that("cv.run works with multiple score functions", {
  cv <- cv.setup(x, y, score=list(mse=mean_se, mae=mean_ae), num_folds = 3, num_iter = 2)
  result <- cv.run(cv, regr)
  expect_equal(dim(result$scores), c(cv$num_iter, cv$num_folds, length(cv$score)) )
  expect_equal(length(result$score.mean), 2)
  expect_equal(length(result$score.sd), 2)
  expect_equal(dim(result$score.iter.mean), c(cv$num_iter, length(cv$score)) )
  
  expect_equal(names(result$score.mean), c("mse", "mae"))
})

test_that("cv.run works with user.score", {
  cv <- cv.setup(x, y, score="user.score", num_folds = 3, num_iter = 2)
  result <- cv.run(cv, regr.score)
  expect_equal(result$Nscores, 1)
  expect_equal(nrow(result$scores), cv$num_iter)
  expect_equal(ncol(result$scores), cv$num_folds)
  expect_equal(dim(result$scores)[3], 1)
  
  expect_equal(length(result$score.mean), 1)
  expect_equal(length(result$score.sd), 1)
  expect_equal(dim(result$score.iter.mean), c(cv$num_iter, length(cv$score)) )
})

test_that("cv.run works with 2 user.score's", {
  cv <- cv.setup(x, y, score="user.score", num_folds = 3, num_iter = 2)
  result <- cv.run(cv, regr.score2)
  expect_equal(result$Nscores, 2)
  expect_equal(nrow(result$scores), cv$num_iter)
  expect_equal(ncol(result$scores), cv$num_folds)
  expect_equal(dim(result$scores)[3], 2)
  
  expect_equal(length(result$score.mean), 2)
  expect_equal(length(result$score.sd), 2)
  expect_equal(dim(result$score.iter.mean), c(cv$num_iter, 2) )

  expect_equal(names(result$score.mean), c("sqerror", "abserror"))
})

