
context("Random search")

f <- function(x,y) -x*x - 0.5*y*y

test_that("random search works", {
  solution <- random_search(f, vars=list("x"=c(-5,5), "y"=c(-5,5)), num_evals=40 )
  expect_equal( solution$stats$num_evals, 40)
})