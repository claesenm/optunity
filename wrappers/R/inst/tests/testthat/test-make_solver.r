
context("make_solver")

test_that("fail if unknown method", {
  expect_error( {make_solver("unknownx")} )
})


############ random search ##############
test_that("random search with num_evals", {
  result = make_solver("random search", num_evals=40)
  expect_equal( result, TRUE)
})

test_that("random search without arguments fails", {
  expect_error( {make_solver("random search")}, regex="Unable to instantiate")
})


############ grid search ##############
test_that("grid search can be made", {
  result = make_solver("grid search", x=c(1,10))
  expect_equal( result, TRUE)
  result = make_solver("grid search")
  expect_equal( result, TRUE)
})


############ nelder-mead ##############
test_that("nelder-mead can be made", {
  result = make_solver("nelder-mead", x=c(1,10))
  expect_equal( result, TRUE)
  result = make_solver("nelder-mead")
  expect_equal( result, TRUE)
})


############ particle swarm ##############
test_that("particle swarm can be made", {
  result = make_solver("particle swarm", num_particles=10, num_generations=5)
  expect_equal( result, TRUE)
})

test_that("particle swarm without arguments fails", {
  expect_error( {make_solver("particle search")}, regex="Unable to instantiate")
})


############ cma-es ##############
#test_that("cma-es can be made", {
#  result = make_solver("cma-es", num_generations=5)
#  expect_equal( result, TRUE)
#})

#test_that("cma-es without arguments fails", {
#  expect_error( {make_solver("particle search")}, regex="Unable to instantiate")
#})
