
cv.setup <- function(x, y=NULL, score, num_folds=5, num_iter=1, 
                     strata=NULL, clusters=NULL,
                     seed=NULL) {
  if ( ! is.matrix(x) && ! is.data.frame(x) && ! inherits(x, "Matrix"))
    stop("x has to be either a matrix, data.frame or of class Matrix.")
  
  if (missing(score))
    stop("Please set score to NULL or provide score function(ytrue, yscore), like score.accuracy for classification or score.neg.mse for regression. score=NULL means f returns score instead of predictions.")
  
  if (!is.list(score) && !is.function(score) && !is.null(score))
    stop("Score has to be NULL, a function or list of scoring functions.")
  
  if (is.list(score)) {
    if (! all(sapply(score, is.function))) {
      stop("score can be a list of scoring functions, but a non-function element was included.")
    }
  }
  
  setup <- list()
  setup$x = x
  setup$y = y
  setup$score = score
  if (is.function(score)) {
    setup$scorename = deparse(substitute(score))
  } else if (is.list(score)) {
    setup$scorename = names(score)
  } else {
    setup$scorename = "user.score"
  }
  
  setup$num_folds = num_folds
  setup$num_iter  = num_iter
  setup$supervised = ! is.null(y)
  if (setup$supervised && nrow(x) != length(y))
    stop( sprintf("Number of rows of x is not equal to the length of y.", nrow(x), length(y)) )
  
  setup$seed = seed
  if ( ! is.null(seed) ) set.seed(seed)
  
  setup$folds = generate_folds(nrow(x), num_folds=num_folds, num_iter=num_iter,
                               strata=strata, clusters=clusters)
  class(setup) <- "cv.setup"
  return(setup)
}


cv.run <- function(setup, f, ...) {
  if ( ! inherits(setup, "cv.setup"))
    stop("Input setup has to be of class 'cv.setup'. Use cv.setup() to create it.")
  
  Nscores = max(1, length(setup$score))
  scores = array(0, dim = c(setup$num_iter, setup$num_folds, Nscores) )
  dimnames(scores)[[1]] <- as.list( sprintf("iter%d", 1:setup$num_iter) )
  dimnames(scores)[[2]] <- as.list( sprintf("fold%d", 1:setup$num_folds) )
  dimnames(scores)[[3]] <- as.list( setup$scorename )
  
  for (iter in 1:setup$num_iter) {
    for (fold in 1:setup$num_folds) {
      itrain = setup$folds[,iter] != fold
      itest  = ! itrain
      xtrain = setup$x[ itrain, ]
      xtest  = setup$x[ itest,  ]
      ytrain = setup$y[ itrain ]
      ytest  = setup$y[ itest  ]
      yhat <- f(xtrain, ytrain, xtest, ytest, ...)
      
      if (is.null(setup$score)) {
        if (length(yhay) > 1) stop("f returned a vector, but should return 1 numeric value (score).")
        if ( ! is.numeric(yhat)) stop("f returned non-numeric value.")
        scores[iter, fold, 1] <- yhat
      } else {
        if (is.list(setup$score)) {
          ## multiple scores
          s <- sapply(setup$score, function(si) si(ytest, yhat))
          if (! is.numeric(s))
            stop("One of the scores returned non-numeric result. Make sure scores always return numeric.")
          scores[iter, fold,] <- s
        } else {
          ## single score
          scores[iter, fold, 1] <- setup$score(ytest, yhat)
        }
      }
    }
  }

  out <- list()
  out$scores     = scores
  
  out$score.mean = aaply(scores, 3, mean)
  out$score.sd   = aaply(scores, 3, sd)
  out$score.iter.mean = matrix(aaply(scores, c(1,3), mean), ncol=Nscores)
  dimnames(out$score.iter.mean) = dimnames(scores)[c(1,3)]

  return( out )
}

cv.grid_search <- function(setup, f, ..., maximize = TRUE, nested = FALSE) {
  args <- list(...)
  if (length(args) == 0)
    stop("Please provide grid for f, like cv.grid_search(setup, f, param1=c(1, 3, 5)).")
  check_cv_args(f, args)
  
  fcv <- function(...) cv.run(setup, f, ...)$score.mean
  
  res <- grid_search(fcv, ..., maximize = maximize)
  return(res)
}

check_cv_args <- function(f, args) {
  if ("" %in% names(args)) {
    stop(sprintf("Positional arguments for parameters (...) are not supported. Please provide named arguments."))
  }
  fargs <- formals(f)
  if ( ! "..." %in% names(fargs)) {
    ## checking if there are args that are not accepted by f
    for (arg in names(args)) {
      if ( ! arg %in% names(fargs) ) {
        stop(sprintf("Argument '%s' supplied but not present in f.", arg))
      }
    }
  }
}

mean.se <- function(ytrue, yhat) {
  mean((ytrue-yhat)^2)
}

mean.ae <- function(ytrue, yhat) {
  mean(abs(ytrue-yhat))
}

accuracy.from.p <- function(ytrue, yprob) {
  if (is.logical(ytrue)) {
    return( mean(ytrue == (yprob > 0.5)) )
  } else if (is.factor(ytrue)) {
    y2 <- ytrue == levels(ytrue)[2]
    return( mean(y2 == (yprob > 0.5) ) )
  } else {
    y2 <- ytrue == max(ytrue)
    return( mean(y2 == (yprob > 0.5) ) )
  }
}

accuracy <- function(ytrue, yhat) {
  mean(ytrue==yhat)
}

auc.roc <- function(ytrue, yscore, decreasing=TRUE, top=1.0) {
  if (length(ytrue) != length(yscore)) {
    stop(sprintf("length of ytrue(%d) should be the same as length of ycore(%d).", 
                 length(true), length(yscore) ))
  }
  tryCatch( {
    pred <- ROCR::prediction(yscore,  ytrue)
    return( performance(pred, "auc")@y.values[[1]] )
  }, error = function(e) return(NA_real_)
  )
}

auc.pr <- function(ytrue, yscore, decreasing=TRUE) {
  perf <- ROCR::performance(ROCR::prediction(yscore, ytrue), "prec", "rec")
  prec <- perf@y.values[[1]]
  if (is.na(prec[1]))
    prec[1] <- 1.0
  rec  <- perf@x.values[[1]]
  aver <- (prec[-1] + prec[-length(prec)]) / 2
  return( sum(base::diff(rec) * aver) )
}

to1or0 <- function(y) {
  if (is.factor(y)) {
    if (length(levels(y)) > 2) stop("Expected two-level factor, got more.")
    return( (0:1)[as.numeric(y)] )
  }
  if (max(y) == 2) return(y-1)
  return(y)
}

early.rie <- function(ytrue, yscore, decreasing=TRUE, alpha=20.0) {
  enrichvs::rie(yscore, to1or0(ytrue), decreasing=decreasing, alpha=alpha)
}

early.bedroc <- function(ytrue, yscore, decreasing=TRUE, alpha=20.0) {
  enrichvs::bedroc(yscore, to1or0(ytrue), decreasing=TRUE, alpha=20.0)
}


