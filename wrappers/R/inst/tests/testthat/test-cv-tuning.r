
context("CV with tuning")

x <- matrix(runif(50*5), 50, 5)
y <- x[,1] + 0.5*x[,2] + 0.1*runif(50)

## linear regression
regr <- function(x, y, xtest, ytest, reg=0) {
  C =  diag(x=reg, ncol(x))
  beta = solve(t(x) %*% x + C, t(x) %*% y)
  xtest %*% beta
}


test_that("cv.grid_search fails with wrong parameter", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  expect_error( {cv.grid_search(cv, regr, noparam = c(1,2) )} )
})


test_that("cv.grid_search fails with wrong size parameter", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  res <- cv.grid_search(cv, regr, reg = c(0, 1e-2, 1e-1), maximize=FALSE )
  expect_equal(res$stats$num_evals, 3)
})

test_that("cv.grid_search works with 2 params", {
  regr <- function(x, y, xtest, ytest, reg, unused) {
    beta = solve(t(x) %*% x + diag(x=reg, ncol(x)), t(x) %*% y)
    xtest %*% beta
  }
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  res <- cv.grid_search(cv, regr, reg = c(0, 1e-2, 1e-1), unused=c(1), maximize=FALSE )
  expect_equal(res$stats$num_evals, 3)
})

#### cv.random_search

test_that("cv.random_search fails with wrong parameter", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  expect_error( {cv.random_search(cv, regr, noparam = c(1,2) )} )
})

test_that("cv.random_search fails with wrong size parameter", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  expect_error( {cv.random_search(cv, regr, reg = c(1, 2, 3), num_evals=6 )} )
})

test_that("cv.random_search works", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  res <- cv.random_search(cv, regr, reg = c(0, 1.0), num_evals=6, maximize=FALSE )
  expect_equal(res$stats$num_evals, 6)
})


#### cv.nelder_mead

test_that("cv.nelder_mead fails with wrong parameter", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  expect_error( {cv.nelder_mead(cv, regr, noparam = c(1,2) )} )
})

test_that("cv.nelder_mead fails with wrong size parameter", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  expect_error( {cv.nelder_mead(cv, regr, reg = c(1, 2), num_evals=6 )} )
})

test_that("cv.nelder_mead works", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  res <- cv.nelder_mead(cv, regr, reg = 1.0, num_evals=6, maximize=FALSE )
  expect_true(res$stats$num_evals <= 6)
})


#### cv.particle_swarm

test_that("cv.particle_swarm fails with wrong parameter", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  expect_error( {cv.particle_swarm(cv, regr, noparam = c(1,2) )} )
})

test_that("cv.particle_swarm fails with wrong size parameter", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  expect_error( {cv.particle_swarm(cv, regr, reg = c(1, 2, 3) )} )
})

test_that("cv.particle_swarm works", {
  cv <- cv.setup(x, y, score=mean_se, num_folds = 3, num_iter = 2)
  res <- cv.particle_swarm(cv, regr, reg = c(0, 1.0), num_particles=2, num_generations=3, maximize=FALSE )
  expect_true(res$stats$num_evals <= 6)
})

#context("Nested cross-validation")
