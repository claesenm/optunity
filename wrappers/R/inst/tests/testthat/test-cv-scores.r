
context("CV scores")

## linear regression
test_that("auc.roc returns NA when 1 value", {
  roc <- auc.roc(factor(1, levels=1:2), 0.5)
  expect_equal( roc, NA_real_ )
})

test_that("auc.roc returns a number when 2 values", {
  roc <- auc.roc(factor(c(1,2), levels=1:2), c(0.5, 0.2) )
  expect_equal( roc, 0 )
})

test_that("auc.pr returns NA when 1 value", {
  roc <- auc.pr(factor(c(1), levels=1:2), c(0.5) )
  expect_equal( roc, NA_real_ )
})

test_that("auc.pr returns a number when only one class", {
  roc <- auc.pr(factor(c(1,1,1), levels=1:2), c(0.5, 0.3, 0.25) )
  expect_equal( roc, NA_real_ )
})

test_that("auc.pr returns a number when more than 1 value", {
  roc <- auc.pr(factor(c(1,2), levels=1:2), c(0.5, 0.3) )
  expect_equal( roc, 0.25 )
})
