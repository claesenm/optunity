
f <- function(x,y) -x*x - 0.5*y*y


context("Grid search")

test_that("grid search works", {
  solution <- grid_search(f, x=c(-5,1,5), y=c(-5,1,5) )
  expect_equal( solution$stats$num_evals, 9)
  expect_equal( solution$optimum, f(1,1) )
})

context("Random search")

test_that("random search works", {
  solution <- random_search(f, box=list(x=c(-5,5), y=c(-5,5)), num_evals=40 )
  expect_equal( solution$stats$num_evals, 40)
})

