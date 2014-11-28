
context("CV scores")

## linear regression
test_that("auc_roc returns NA when 1 value", {
  roc <- auc_roc(factor(1, levels=1:2), 0.5)
  expect_equal( roc, NA_real_ )
})

test_that("auc_roc returns a number when 2 values", {
  roc <- auc_roc(factor(c(1,2), levels=1:2), c(0.5, 0.2) )
  expect_equal( roc, 0 )
})

test_that("auc_pr returns NA when 1 value", {
  roc <- auc_pr(factor(c(1), levels=1:2), c(0.5) )
  expect_equal( roc, NA_real_ )
})

test_that("auc_pr returns a number when only one class", {
  roc <- auc_pr(factor(c(1,1,1), levels=1:2), c(0.5, 0.3, 0.25) )
  expect_equal( roc, NA_real_ )
})

test_that("auc_pr returns a number when more than 1 value", {
  roc <- auc_pr(factor(c(1,2), levels=1:2), c(0.5, 0.3) )
  expect_equal( roc, 0.25 )
})
