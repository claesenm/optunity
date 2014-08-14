
f <- function(x,y) -x*x - 0.5*y*y


context("Grid search")

test_that("grid_search works", {
  solution <- grid_search(f, x=c(-5,1,5), y=c(-5,1,5) )
  expect_equal( solution$stats$num_evals, 9)
  expect_equal( solution$optimum, f(1,1) )
})

context("Random search")

test_that("random_search works", {
  solution <- random_search(f, x=c(-5,5), y=c(-5,5), num_evals=40 )
  expect_equal( solution$stats$num_evals, 40)
})

context("Nelder-mead")

test_that("nelder_mead works", {
  solution <- nelder_mead(f, x=5, y=-5, num_evals=40 )
  expect_true( solution$stats$num_evals < 40)
})

context("Particle swarm")

test_that("particle_swarm works", {
  solution <- particle_swarm(f, x=c(-5,5), y=c(-5,5), num_particles=10, num_generations=4 )
  expect_equal( solution$stats$num_evals, 40)
})
