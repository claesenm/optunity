
manual <- function(solver_name=''){
    cons <- launch()
    on.exit(close_pipes(cons))

    msg <- list(manual = solver_name)
    send(cons$r2py, msg)

    content <- receive(cons$py2r)
    cat(content$manual, sep="\n")
    return (content$solver_names)
}

make_solver <- function(solver_name, ...){
  cons <- launch()
  on.exit(close_pipes(cons))
  
  # create solver config
  cfg <- c(list(), solver_name = solver_name, ...)
  
  msg <- list(make_solver = cfg)
  send(cons$r2py, msg)
  
  reply <- receive(cons$py2r)
  return(TRUE)
}

generate_folds <- function(num_instances, num_folds=10,
                           num_iter=1, strata=list(),
                           clusters=list()){
    cons <- launch()
    on.exit(close_pipes(cons))

    # create config for generating folds
    cfg <- list(num_instances = num_instances,
                num_folds = num_folds, num_iter = num_iter)
    if (length(strata) > 0) cfg$strata <- strata
    if (length(clusters) > 0) cfg$clusters <- clusters

    msg <- list(generate_folds = cfg)
    send(cons$r2py, msg)

    reply <- receive(cons$py2r)

    folds <- array(0, dim=c(num_instances, num_iter))
    for (iter in 1:num_iter){
        for (fold in 1:num_folds){
            folds[1+reply$folds[[iter]][[fold]], iter] <- fold
        }
    }
    return (folds)
}

random_search <- function(f, ..., maximize  = TRUE, num_evals = 50) {
  # {"optimize" : {"max_evals": 0}, "solver": {"solver_name" : "random search", "num_evals": 5, "x":[0,10]} }
  conf <- list(...)
  if (length(conf) == 0) stop("Please provide bounds for each variable of f, e.g. random_search(f, var1=c(-5, 5), var2=c(-1,1)).")
  conf$num_evals = num_evals
  return( optimize2(f, solver_name="random search", maximize=maximize, solver_config = conf) )
}

grid_search <- function(f, ..., maximize  = TRUE) {
  # {"optimize" : {"max_evals": 0}, "solver": {"solver_name" : "grid search", "x":[0,10]}}
  args <- list(...)
  if ( length(args) == 0) stop("Please provide grid for f, e.g., grid_search(f, var1=c(1, 3, 5)).")
  return( optimize2(f, solver_name="grid search", maximize=maximize, solver_config = args) )
}

nelder_mead <- function(f, ..., num_evals = 50, maximize = TRUE) {
  # {"optimize" : {"max_evals": 0}, "solver": {"solver_name" : "nelder-mead", "x":2}}
  args <- list(...)
  if ( length(args) == 0) stop("Please provide initial value for f, e.g., nelder_mead(f, var1=1, var2=3).")
  args$max_iter = as.integer(num_evals / 2) - 1
  return( optimize2(f, solver_name="nelder-mead", maximize=maximize, solver_config = args) )
}

particle_swarm <- function(f, ..., num_particles=10, num_generations=5, maximize = TRUE) {
  # {"optimize" : {"max_evals": 0}, "solver": {"solver_name" : "particle swarm", "x":[2, 6]}}
  args <- list(...)
  if (length(args) == 0) stop("Please provide bounds for each variable of f, e.g. particle_swarm(f, var1=c(-5, 5), var2=c(0.1, 10)).")
  args$num_particles   = num_particles
  args$num_generations = num_generations
  return( optimize2(f, solver_name="particle swarm", maximize=maximize, solver_config = args) )
}

optimize2 <- function(f,
                      solver_name,
                      solver_config = list(),
                      constraints = NULL,
                      maximize    = TRUE,
                      max_evals   = 0,
                      call_log    = NULL,
                      return_call_log = FALSE,
                      default = NULL){
    if ( ! is.logical(return_call_log)) stop("Input 'return_call_log' has to be TRUE or FALSE.")
    if ( ! is.logical(maximize))        stop("Input 'maximize' has to be TRUE or FALSE.")

    cons <- launch()
    on.exit(close_pipes(cons))

    msg <- list(
      optimize = list(max_evals = max_evals, maximize=maximize),
      solver   = c( list(solver_name = solver_name),
                    solver_config )
                    #return_call_log = return_call_log)
    )

    if (!is.null(call_log)) msg$call_log <- call_log
    if (!is.null(constraints)) msg$constraints <- constraints
    if (!is.null(default)) msg$default <- default

    send(cons$r2py, msg)
    repeat{
        reply <- receive(cons$py2r)
        if ("solution" %in% names(reply)) break

        if (is.null(names(reply))) {
          ## vector evaluation
          values <- simplify2array(
            lapply(reply, function(param) do.call(f, param))
          )
          if ( ! is.vector(values) || ! is.numeric(values) ) {
            problem <- which( ! sapply(values, is.numeric) | sapply(values, length) != 1)
            i <- problem[1]
            stop(sprintf("Call f(%s) gave output '%s'. Function f has to return a single numeric value."), 
                 toString( reply[[i]] ),
                 toString( values[[i]] )
            )
          }
          ## returning results of vector evaluation
          send(cons$r2py, list(values=values))
        } else {
          ## single evaluation
          value <- do.call(f, reply)
          send(cons$r2py, list(value=value))
        }
        
    }
    return (reply)
}
