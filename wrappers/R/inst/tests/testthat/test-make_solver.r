
context("make_solver")

test_that("random search can be made", {
  result = make_solver("random search", num_evals=40)
  expect_equal( result, TRUE)
  result = make_solver("random search")
  expect_message( result, regex="Unable to instantiate")
})

test_that("random search works", {
  solution <- random_search(f, vars=list("x"=c(-5,5), "y"=c(-5,5)), num_evals=40 )
  expect_equal( solution$stats$num_evals, 40)
})
