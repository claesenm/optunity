
manual <- function(solver_name=''){
    cons <- launch()
    on.exit(close_pipes(cons))

    msg <- list(manual = TRUE)
    if (solver_name != '') msg$solver <- solver_name
    send(cons$r2py, msg)

    content <- receive(cons$py2r)
    cat(content$manual, sep="\n")
    return (content$solver_names)
}

generate_folds <- function(num_instances, num_folds=10,
                           num_iter=1, strata=list(),
                           clusters=list()){
    cons <- launch()
    on.exit(close_pipes(cons))

    # create solver config
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

optimize2 <- function(solver_name, solver_config, f,
                  constraints = NULL, call_log = NULL,
                  return_call_log = FALSE,
                  default = NULL){
    cons <- launch()
    on.exit(close_pipes(cons))

    msg <- list(solver=list(solver_name=solver_name, config=solver_config,
               return_call_log = return_call_log)
    if (!is.null(call_log)) msg$call_log <- call_log
    if (!is.null(constraints)) msg$constraints <- constraints
    if (!is.null(default)) msg$default <- default

    send(cons$r2py, msg)
    repeat{
        reply <- receive(cons$py2r)
        if ("solution" %in% names(reply)) break

        value <- do.call(f, reply)
        send(cons$r2py, list(value=value))
    }
    return (reply)
}
