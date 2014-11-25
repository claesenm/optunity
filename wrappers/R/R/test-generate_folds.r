context("Generate Folds")

test_that("Basic generate folds works", {
  folds = generate_folds(num_instances = 20, num_folds = 5)
  expect_equal(dim(folds), c(20, 1))
  expect_equal(max(folds), 5)
  expect_equal(
    array( table(folds[,1]) ),
    array( c(4,4,4,4,4) )
  )

  folds = generate_folds(num_instances = 20, num_folds = 5, num_iter=2)
  expect_equal(dim(folds), c(20, 2))
  expect_equal(max(folds), 5)
})

test_that("Generate folds with strata works", {
  folds = generate_folds(num_instances = 10, num_folds = 5, strata = list(1:5, 6:10))
  expect_equal(dim(folds), c(10, 1))
  expect_equal(max(folds), 5)
})

