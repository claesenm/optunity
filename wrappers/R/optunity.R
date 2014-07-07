source("comm.R")

manual <- function(solver_name=''){
    # kludgy RAII, since R doesn't support it properly
    cons <- launch()
    on.exit(close(cons$py2r))
    on.exit(close(cons$r2py))
    on.exit(system(paste('rm -f ',py2r_name,sep='')))

    msg <- c()
    msg$manual <- TRUE
    if (solver_name != '') msg$solver <- solver_name
    send(cons$r2py, msg)

    content <- receive(cons$py2r)
    cat(content$manual, sep="\n")
    return (content$solver_names)
}

generate_folds <- function(num_instances, num_folds=10,
                           num_iter=1, strata=list(),
                           clusters=list()){
    # kludgy RAII, since R doesn't support it properly
    cons <- launch()
    on.exit(close(cons$py2r))
    on.exit(close(cons$r2py))
    on.exit(system(paste('rm -f ',py2r_name,sep='')))

    # create solver config
    cfg <- list(num_instances = num_instances, 
                num_folds = num_folds, num_iter = num_iter)
    if (length(strata) > 0) cfg$strata <- strata
    if (length(clusters) > 0) cfg$clusters <- clusters

    msg <- list(generate_folds = cfg)
    send(cons$r2py, msg)

    reply <- receive(cons$py2r)
    return (reply$folds)
}
