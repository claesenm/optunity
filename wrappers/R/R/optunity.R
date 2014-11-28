#' Displays manuals for optimization methods in Optunity package.
#'
#' @param solver_name solver name, or an empty string for a of all available solvers
#' @return Solver name(s) or list of available solvers if solver_name=''. Also prints out manual for the solver.
#' @seealso \code{\link{optimize2}} low level function for running different solvers
#' @export
#' @examples
#' manual('')
#' manual('particle swarm')
manual <- function(solver_name=''){
    cons <- launch()
    on.exit(close_pipes(cons))

    msg <- list(manual = solver_name)
    send(cons$socket, msg)

    content <- receive(cons$socket)
    cat(content$manual, sep="\n")
    return (content$solver_names)
}

#' Tests if solver can be made with supplied arguments. Low level API.
#'
#' @param solver_name solver name
#' @param ...         parameters passed to making the solver
#' @return TRUE if succeeds, otherwise throws error
#' @seealso \code{\link{manual}} for list of solvers and their arguments
#' @export
#' @examples
#' make_solver('random search', num_evals = 30)
make_solver <- function(solver_name, ...){
  cons <- launch()
  on.exit(close_pipes(cons))
  
  # create solver config
  cfg <- c(list(), solver_name = solver_name, ...)
  
  msg <- list(make_solver = cfg)
  send(cons$socket, msg)
  
  reply <- receive(cons$socket)
  return(TRUE)
}

#' Generates folds for cross-validation.
#'
#' @param num_instances number of samples
#' @param num_folds     number of folds
#' @param num_iter      number of iterations (for doing repeated cross-validation)
#' @param strata        (optional) list of strata, each strata is a list of samples ids that will be stratified (balanced) among the folds
#' @param clusters      (optional) list of clusters, each cluster is a list of samples that always goes to the same fold
#' @return matrix of folds, size: num_instances x num_iter
#' @seealso \code{\link{cv.setup}} for full cross-validation approach
#' @export
#' @examples
#' folds1 = generate_folds(num_instances = 20, num_folds = 5)
#' folds2 = generate_folds(num_instances = 20, num_folds = 5, num_iter=2)
#' ## stratified folds
#' folds3 = generate_folds(num_instances = 100, strata = list(1:50, 51:100))
generate_folds <- function(num_instances, num_folds=5,
                           num_iter=1, strata=NULL,
                           clusters=NULL){
  if ( ! is.numeric(num_iter)) stop("num_iter has to be numeric.")
  if ( ! is.numeric(num_folds)) stop("num_folds has to be numeric.")
  
  cons <- launch()
  on.exit(close_pipes(cons))
  
  # create config for generating folds
  cfg <- list(num_instances = num_instances,
              num_folds = num_folds, num_iter = num_iter)
  if (length(strata) > 0) cfg$strata <- strata
  if (length(clusters) > 0) cfg$clusters <- clusters
  
  msg <- list(generate_folds = cfg)
  send(cons$socket, msg)
  
  reply <- receive(cons$socket)
  
  folds <- array(0, dim=c(num_instances, num_iter))
  for (iter in 1:num_iter){
    for (fold in 1:num_folds){
      folds[1+reply$folds[[iter]][[fold]], iter] <- fold
    }
  }
  return (folds)
}

#' Finds optimum for function using random search.
#'
#' @param f         function to be optimized
#' @param ...       box constraints of form x = c(-5, 5) where x is input parameter to f
#' @param maximize  whether to maximize or minimize
#' @param num_evals maximum number of evaluations of f
#' @return solution and details
#' @seealso \code{\link{grid_search}}, \code{\link{nelder_mead}} and \code{\link{particle_swarm}} for other high level methods for optimization.
#' @export
#' @examples
#' f <- function(x,y) -x*x - 0.5*y*y
#' opt <- random_search(f, x=c(-5,5), y=c(-5,5), num_evals=40 )
#' ## solution found
#' opt$solution
#' ## value of f at the solution
#' opt$optimum
random_search <- function(f, ..., maximize  = TRUE, num_evals = 50) {
  # {"optimize" : {"max_evals": 0}, "solver": {"solver_name" : "random search", "num_evals": 5, "x":[0,10]} }
  args <- list(...)
  check_box(args, "random_search")
  check_args(f, args)
  args$num_evals = num_evals
  return( optimize2(f, solver_name="random search", maximize=maximize, solver_config = args) )
}

#' Finds optimum for function using grid search.
#'
#' @param f         function to be optimized
#' @param ...       grid in the form x = c(-5, -1, 1, 5) where x is input parameter to f
#' @param maximize  whether to maximize or minimize
#' @return solution and details
#' @seealso \code{\link{random_search}}, \code{\link{nelder_mead}} and \code{\link{particle_swarm}} for other high level methods for optimization.
#' @export
#' @examples
#' f <- function(x,y) -x*x - 0.5*y*y
#' opt <- grid_search(f, x=seq(-5, 5, 2), y=seq(-5, 5, 2))
#' ## solution found
#' opt$solution
#' ## value of f at the solution
#' opt$optimum
grid_search <- function(f, ..., maximize  = TRUE) {
  # {"optimize" : {"max_evals": 0}, "solver": {"solver_name" : "grid search", "x":[0,10]}}
  args <- list(...)
  if ( length(args) == 0) stop("Please provide grid for f, like grid_search(f, var1=c(1, 3, 5)).")
  check_args(f, args, "grid", "c(0.01, 0.1, 1.0)")
  for (a in names(args)) {
    if (length(args[[a]]) == 1) {
      args[[a]] = list(unname(args[[a]]))
    }
  }
  return( optimize2(f, solver_name="grid search", maximize=maximize, solver_config = args) )
}

#' Finds optimum for function using Nelder-Mead.
#'
#' @param f         function to be optimized
#' @param ...       starting point in the form x = 5, where x is input parameter to f
#' @param maximize  whether to maximize or minimize
#' @param num_evals maximum number of evaluations of f
#' @return solution and details
#' @details Nelder-Mead is efficient if f is convex, otherwise it can easily get stuck into local minima.
#' @seealso \code{\link{grid_search}}, \code{\link{random_search}}, and \code{\link{particle_swarm}} for other high level methods for optimization.
#' @export
#' @examples
#' f <- function(x,y) -x*x - 0.5*y*y
#' opt <- nelder_mead(f, x=5, y=5)
#' ## solution found
#' opt$solution
#' ## value of f at the solution
#' opt$optimum
nelder_mead <- function(f, ..., num_evals = 50, maximize = TRUE) {
  # {"optimize" : {"max_evals": 0}, "solver": {"solver_name" : "nelder-mead", "x":2}}
  args <- list(...)
  if (length(args) == 0) stop("Please provide initial value for f, like nelder_mead(f, var1=1, var2=3).")
  if (any(sapply(args, length) != 1)) stop("Please provide initial value(s) for f, as scalars, like nelder_mead(f, var1=1, var2=3).")
  check_args(f, args, "initial values", "2.5")
  args$max_iter = as.integer(num_evals / 2) - 1
  return( optimize2(f, solver_name="nelder-mead", maximize=maximize, solver_config = args) )
}

#' Finds optimum for function using Particle Swarm.
#'
#' @param f         function to be optimized
#' @param ...       box constraints of form x = c(-5, 5) where x is input parameter to f
#' @param num_particles   number of particles
#' @param num_generations number of generations
#' @param maximize  whether to maximize or minimize
#' @return solution and details
#' @seealso \code{\link{grid_search}}, \code{\link{random_search}}, and \code{\link{particle_swarm}} for other high level methods for optimization.
#' @export
#' @examples
#' f <- function(x,y) -x*x - 0.5*y*y
#' opt <- particle_swarm(f, x=c(-5, 5), y=c(-5, 5) )
#' ## solution found
#' opt$solution
#' ## value of f at the solution
#' opt$optimum
particle_swarm <- function(f, ..., num_particles=5, num_generations=10, maximize = TRUE) {
  # {"optimize" : {"max_evals": 0}, "solver": {"solver_name" : "particle swarm", "x":[2, 6]}}
  args <- list(...)
  check_box(args, "particle_swarm")
  check_args(f, args)
  args$num_particles   = num_particles
  args$num_generations = num_generations
  return( optimize2(f, solver_name="particle swarm", maximize=maximize, solver_config = args) )
}

cma_es <- function(f, ..., num_generations=10, maximize = TRUE) {
  # {"optimize" : {"max_evals": 0}, "solver": {"solver_name" : "cma-es", "num_generations":5, "x":[2, 6]}}
  args <- list(...)
  check_args(f, args, "initial values", "2.5")
  args$num_generations = num_generations
  return( optimize2(f, solver_name="cma-es", maximize=maximize, solver_config = args) )
}

## check all required args are supplied
check_args <- function(f, args, vartype = "bounds", example = "c(0.1, 10)") {
  if ("" %in% names(args)) {
    stop(sprintf("Positional arguments for ... are not supported. Please provide named arguments, like var1=%s.", example))
  }
  fargs <- formals(f)
  fargs.req <- names(fargs)[ sapply(fargs, is.symbol) ]
  missing <- fargs.req[ ! (fargs.req %in% c("...",names(args)) ) ]
  if (length(missing) > 0) {
    stop(sprintf("Missing %s for argument '%s' for the target function. ", vartype, missing))
  }
}

## check box constraints
check_box <- function(args, methodName) {
  if (length(args) == 0) stop(
    sprintf("Please provide bounds for each variable of f, like %s(f, var1=c(-5, 5), var2=c(0.1, 10)).", methodName)
  )
  u <- which(sapply(args, length) != 2)
  if (length(u) > 0) {
    stop(sprintf("Bounds for variable '%s' has %d values. It should be two values: c(lower, upper). ",
                 names(args)[u],
                 length(args[u])
    ))
  }
  ## bounds are numeric
  u <- which( ! sapply(args, is.numeric))
  if (length(u) > 0) stop(sprintf("Bounds for variable '%s' have to be numeric. ", names(args)[u] ))
  
  ## check bounds are ordered correctly
  u <- which( ! sapply(args, function(x) x[1] <= x[2]))
  if (length(u) > 0) {
    stop(sprintf("Bounds for variable '%s' has lower bound bigger (%f) than upper bound (%f). ",
                 names(args)[u],
                 args[u][1],
                 args[u][2]
    ))
  }
}

#' Low level function to call any optimizer with custom configuration.
#'
#' @param f             function to be optimized
#' @param solver_name   name of the solver (see manual() for available solvers)
#' @param solver_config list of configuration options for the solver
#' @param constraints   list of constraints, if needed
#' @param maximize      whether to maximize (TRUE) or minimize (FALSE)
#' @param max_evals     number of evaluations to perform
#' @param call_log      currently not used.
#' @param default       default value given to solvers when constraints are violated
#' @return result of optimization
#' @seealso \code{\link{grid_search}}, \code{\link{random_search}}, \code{\link{nelder_mead}} and \code{\link{particle_swarm}} for high level methods for optimization.
#' @export
#' @examples
#' ## function to be minimized
#' f <- function(x,y,z) { (x-1)^2 + (y-1)^2 + (z-2)^2 }
#' ## bounds for parameters
#' args = list(x = c(-10, 10), y = c(-10, 10), z = c(-10, 10))
#' ## particle swarm setup
#' args$num_particles   = 8
#' args$num_generations = 10
#'
#' optimize2(f, solver_name="particle swarm", maximize=FALSE, solver_config =args )
optimize2 <- function(f,
                      solver_name,
                      solver_config = list(),
                      constraints = NULL,
                      maximize    = TRUE,
                      max_evals   = 0,
                      call_log    = NULL,
                      default = NULL){
    if ( ! is.logical(maximize))        stop("Input 'maximize' has to be TRUE or FALSE.")

    cons <- launch()
    on.exit(close_pipes(cons))

    msg <- list(
      optimize = list(max_evals = max_evals, maximize=maximize),
      solver   = c( list(solver_name = solver_name),
                    solver_config )
    )

    if (!is.null(call_log)) msg$call_log <- call_log
    if (!is.null(constraints)) msg$constraints <- constraints
    if (!is.null(default)) msg$default <- default

    send(cons$socket, msg)
    repeat{
        reply <- receive(cons$socket)
        if ("solution" %in% names(reply)) break

        if (is.null(names(reply))) {
          ## vector evaluation
          values <- simplify2array(
            lapply(reply, function(param) do.call(f, param))
          )
          if ( ! is.vector(values) || ! is.numeric(values) ) {
            problem <- which( ! sapply(values, is.numeric) | sapply(values, length) != 1)
            i <- problem[1]
            stop(sprintf("Call f(%s) gave output '%s'. Function f has to return a single numeric value.",
                         toString( reply[[i]] ),
                         toString( values[[i]] )
            ))
          }
          ## returning results of vector evaluation
          send(cons$socket, list(values=values))
        } else {
          ## single evaluation
          value <- do.call(f, reply)
          send(cons$socket, list(value=value))
        }
        
    }
    return (reply)
}
