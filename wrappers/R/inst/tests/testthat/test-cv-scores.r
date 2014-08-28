
context("CV scores")

## linear regression
test_that("auc.roc returns NA when 1 value", {
  roc <- auc.roc(factor(1, levels=1:2), 0.5)
  expect_equal( roc, NA_real_ )
})

test_that("auc.roc returns a number when 2 values", {
  roc <- auc.roc(factor(c(1,2), levels=1:2), c(0.5,0.2) )
  expect_equal( roc, 0 )
})

