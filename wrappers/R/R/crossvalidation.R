
#' Creates cross-validation setup
#'
#' @param x     data matrix
#' @param y     output labels, useful for supervised setup
#' @param score score function, like mean_se, or a list of functions. Set it to 'user.score' if evaluated functions (in cv.run, cv.particle_swarm) will also perform scoring.
#' @inheritParams generate_folds
#' @param seed  set seed for the random generator for generating folds
#' @return object of class 'cv.setup' to be used for cv.run and optimization methods, like cv.particle_swarm
#' @seealso \code{\link{cv.run}} for running cross-validation and \code{\link{cv.particle_swarm}} for finding optimal parameters
#' @details cv.setup is used to make a setup by passing in your data as x and y (later is optional). The cv.setup object will define how the data will be partitioned and the score(s) computed in in the later commands.
#' @export
#' @examples
#' ## data
#' x <- matrix(runif(50*40), 50, 40)
#' y <- x[,1] + 0.5*x[,2] + 0.1*runif(50)
#'
#' ## ridge regression
#' regr <- function(x, y, xtest, ytest, reg=0) {
#'     C =  diag(x=reg, ncol(x))
#'     beta = solve(t(x) %*% x + C, t(x) %*% y)
#'     ## make predictions for xtest
#'     xtest %*% beta
#' }
#'
#' ## compute mean squared error with CV with 2 repeats (iterations) and 10 folds
#' cv <- cv.setup(x, y, score=mean_se, num_folds = 10, num_iter = 2)
#' result <- cv.run(cv, regr)
#' result$score.mean
#'
#' ## compute both mean squared error and absolute error with CV
#' cv <- cv.setup(x, y, score=list(mse=mean_se, mae=mean_ae), num_folds = 10, num_iter = 2)
#' result <- cv.run(cv, regr)
#' result$score.mean
#'
#'
#' ###### Example of user.score #######
#' ## linear regression with own scoring function
#' regr.score <- function(x, y, xtest, ytest, reg=0) {
#'   C =  diag(x=reg, ncol(x))
#'   beta = solve(t(x) %*% x + C, t(x) %*% y)
#'   sum((xtest %*% beta - ytest)^2)
#' }
#' 
#' cv <- cv.setup(x, y, score="user.score", num_folds = 3, num_iter = 2)
#' result <- cv.run(cv, regr.score, reg = 0.1)
#' result$score.mean
cv.setup <- function(x, y=NULL, score, num_folds=5, num_iter=1, 
                     strata=NULL, clusters=NULL,
                     seed=NULL) {
  if ( ! is.matrix(x) && ! is.data.frame(x) && ! inherits(x, "Matrix"))
    stop("x has to be either a matrix, data.frame or of class Matrix.")
  
  if (missing(score))
    stop("Please set score to 'user.score' or provide score function(ytrue, yscore), like accuracy for classification or mean_se for regression. If score='user.score' then f has to return score(s) instead of predictions.")
  
  if (!is.list(score) && !is.function(score) && score != 'user.score')
    stop("Score has to be 'user.score', a function or list of scoring functions.")
  
  if (is.list(score)) {
    if (! all(sapply(score, is.function))) {
      notfunc = score[sapply(score, is.function) == FALSE][1]
      stop(sprintf(
        "score can be a list of scoring functions, but a non-function element was included: %s.", notfunc))
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
    setup$scorename = ""
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


#' Runs cross-validation on predict-train function
#'
#' @param setup cv settings created by cv.setup
#' @param f     function that trains a model and makes prediction. Its 4 first inputs have to be xtrain, ytrain, xtest, ytest. Should return predictions of xtest or final score, depending on cv setup, see details.
#' @param ...   additional parameters that will be passed to f
#' @return cross-validation scores for method f 
#' @seealso \code{\link{cv.setup}} for creating cv.setup object and \code{\link{cv.particle_swarm}} for finding optimal parameters
#' @details 
#' There are two modes for f, depending on cv.setup's score parameter, if:
#' \itemize{
#'   \item{score=a_score_function}{ f should perform train-predict: train model on xtrain and ytrain, then return predictions on xtest}
#'   \item{score='user.score'}{ f should perform train-predict-score: train model on xtrain and ytrain, then make predictions on xtest, finally return score based on ytest.}
#' }
#' @export
#' @examples
#' ## data
#' x <- matrix(runif(50*40), 50, 40)
#' y <- x[,1] + 0.5*x[,2] + 0.1*runif(50)
#'
#' ## ridge regression
#' regr <- function(x, y, xtest, ytest, reg=0) {
#'     C =  diag(x=reg, ncol(x))
#'     ## train model
#'     beta = solve(t(x) %*% x + C, t(x) %*% y)
#'     ## make predictions for xtest
#'     xtest %*% beta
#' }
#' cv <- cv.setup(x, y, score=mean_se, num_folds = 5, num_iter = 2)
#' ## checking accuracy with different regularization
#' result1 <- cv.run(cv, regr, reg = 0.1)
#' result2 <- cv.run(cv, regr, reg = 1.0)
cv.run <- function(setup, f, ...) {
  if ( ! inherits(setup, "cv.setup"))
    stop("Input setup has to be of class 'cv.setup'. Use cv.setup() to create it.")
  
  Nscores = max(1, length(setup$score))
  scores = array(0, dim = c(setup$num_iter, setup$num_folds, Nscores) )

  first = TRUE
  user.score = is.character(setup$score) && setup$score[1] == "user.score"
  
  for (iter in 1:setup$num_iter) {
    for (fold in 1:setup$num_folds) {
      itrain = setup$folds[,iter] != fold
      itest  = ! itrain
      xtrain = setup$x[ itrain, ]
      xtest  = setup$x[ itest,  ]
      ytrain = setup$y[ itrain ]
      ytest  = setup$y[ itest  ]
      yhat <- f(xtrain, ytrain, xtest, ytest, ...)
      
      if (user.score) {
        if ( ! is.numeric(yhat)) stop("f returned non-numeric value.")
        if (first) {
          ## first scoring, computing resulting array size
          Nscores = length(yhat)
          scores  = array(0, dim = c(setup$num_iter, setup$num_folds, Nscores) )
          first   = FALSE
          if ( ! is.null(names(yhat)) ) {
            setup$scorename = names(yhat)
          } else {
            setup$scorename = paste0("user.score", 1:length(yhat))
          }
        }
        if (length(yhat) != Nscores) {
          stop(sprintf("f returned a vector of %d, but should return %d numeric value (score).", length(yhat), dim(scores)[3] ))
        } else {
          scores[iter, fold, ] <- yhat
        }
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
      first = FALSE
    }
  }
  
  dimnames(scores)[[1]] <- as.list( sprintf("iter%d", 1:setup$num_iter) )
  dimnames(scores)[[2]] <- as.list( sprintf("fold%d", 1:setup$num_folds) )
  dimnames(scores)[[3]] <- as.list( setup$scorename )

  out <- list()
  out$scores     = scores
  out$Nscores    = Nscores
  
  out$score.mean = aaply(scores, 3, mean, na.rm = TRUE)
  out$score.sd   = aaply(scores, 3, sd, na.rm = TRUE)
  out$score.iter.mean = matrix(aaply(scores, c(1,3), mean, na.rm = TRUE), ncol=Nscores)
  dimnames(out$score.iter.mean) = dimnames(scores)[c(1,3)]

  return( out )
}

#' Grid search to find parameters giving best cross-validated score
#' @inheritParams cv.run
#' @param ...       grid in the form \code{logreg = -2:3} where logreg is input parameter to f
#' @param maximize  whether to maximize or minimize the score (first of the scores if cv.setup has several scores)
#' @param nested    whether to perform nested CV
#' @seealso \code{\link{cv.run}} for how to define learn-predict function f.
#' @examples
#' x <- matrix(runif(50*5), 50, 40)
#' y <- x[,1] + 0.5*x[,2] + 0.1*runif(50)
#' 
#' ## linear regression
#' regr <- function(x, y, xtest, ytest, logreg=0) {
#'   C =  diag(x = 10^logreg, ncol(x))
#'   beta = solve(t(x) %*% x + C, t(x) %*% y)
#'   xtest %*% beta
#' }
#' cv <- cv.setup(x, y, score=mean_se, num_folds = 5, num_iter = 2)
#' res <- cv.grid_search(cv, regr, logreg = -2:3, maximize=FALSE )
cv.grid_search <- function(setup, f, ..., maximize = TRUE, nested = FALSE) {
  if (nested) stop("Nested cross-validation is not yet supported.")
  args <- list(...)
  if (length(args) == 0)
    stop("Please provide grid for f, like cv.grid_search(setup, f, param1=c(1, 3, 5)).")
  check_cv_args(f, args)
  
  fcv <- function(...) cv.run(setup, f, ...)$score.mean
  
  res <- grid_search(fcv, ..., maximize = maximize)
  return(res)
}

#' Random search to find parameters giving best cross-validated score
#' @inheritParams cv.run
#' @param ...       box constraints for parameters in the form \code{logreg = c(-2, 5)} where logreg is input parameter to f
#' @param maximize  whether to maximize or minimize the score (first of the scores if cv.setup has several scores)
#' @param num_evals maximum number of parameter evaluations
#' @param nested    whether to perform nested CV
#' @seealso \code{\link{cv.run}} for how to define learn-predict function f.
#' @examples
#' x <- matrix(runif(50*5), 50, 40)
#' y <- x[,1] + 0.5*x[,2] + 0.1*runif(50)
#' 
#' ## linear regression
#' regr <- function(x, y, xtest, ytest, logreg=0) {
#'   C =  diag(x = 10^logreg, ncol(x))
#'   beta = solve(t(x) %*% x + C, t(x) %*% y)
#'   xtest %*% beta
#' }
#' cv <- cv.setup(x, y, score=mean_se, num_folds = 5, num_iter = 2)
#' res <- cv.random_search(cv, regr, logreg = c(-2, 5), num_evals = 10, maximize=FALSE)
cv.random_search <- function(setup, f, ..., maximize = TRUE, num_evals = 50, nested = FALSE) {
  if (nested) stop("Nested cross-validation is not yet supported.")
  args <- list(...)
  if (length(args) == 0)
    stop("Please provide parameter intervals for f, like cv.random_search(setup, f, param1=c(0, 20)).")
  check_cv_args(f, args)
  
  fcv <- function(...) cv.run(setup, f, ...)$score.mean
  
  res <- random_search(fcv, ..., maximize = maximize, num_evals = num_evals)
  return(res)
}

#' Nelder-Mead optimization to find parameters giving best cross-validated score
#' @inheritParams cv.run
#' @param ...       starting point for the parameter search, in the form \code{logreg = 1} where logreg is input for f
#' @param maximize  whether to maximize or minimize the score (first of the scores if cv.setup has several scores)
#' @param num_evals maximum number of parameter evaluations
#' @param nested    whether to perform nested CV
#' @seealso \code{\link{cv.run}} for how to define learn-predict function f.
#' @examples
#' x <- matrix(runif(50*5), 50, 40)
#' y <- x[,1] + 0.5*x[,2] + 0.1*runif(50)
#' 
#' ## linear regression
#' regr <- function(x, y, xtest, ytest, logreg=0) {
#'   C =  diag(x = 10^logreg, ncol(x))
#'   beta = solve(t(x) %*% x + C, t(x) %*% y)
#'   xtest %*% beta
#' }
#' cv <- cv.setup(x, y, score=mean_se, num_folds = 5, num_iter = 2)
#' res <- cv.nelder_mead(cv, regr, logreg = 2, num_evals = 10, maximize=FALSE)
cv.nelder_mead <- function(setup, f, ..., maximize = TRUE, num_evals = 50, nested = FALSE) {
  if (nested) stop("Nested cross-validation is not yet supported.")
  args <- list(...)
  if (length(args) == 0)
    stop("Please provide parameter initial values for f, like cv.nelder_mead(setup, f, param1=2)).")
  check_cv_args(f, args)
  
  fcv <- function(...) cv.run(setup, f, ...)$score.mean
  
  res <- nelder_mead(fcv, ..., maximize = maximize, num_evals = num_evals)
  return(res)
}

#' Particle swarm optimization to find parameters giving best cross-validated score
#' @inheritParams cv.run
#' @param ...       box constraints for parameters in the form \code{logreg = c(-2, 5)} where logreg is input parameter to f
#' @param maximize  whether to maximize or minimize the score (first of the scores if cv.setup has several scores)
#' @param num_particles   number of particles (setting for particle swarm)
#' @param num_generations number of generations (setting for particle swarm)
#' @param nested          whether to perform nested CV
#' @seealso \code{\link{cv.run}} for how to define learn-predict function f.
#' @examples
#' x <- matrix(runif(50*5), 50, 40)
#' y <- x[,1] + 0.5*x[,2] + 0.1*runif(50)
#' 
#' ## linear regression
#' regr <- function(x, y, xtest, ytest, logreg=0) {
#'   C =  diag(x = 10^logreg, ncol(x))
#'   beta = solve(t(x) %*% x + C, t(x) %*% y)
#'   xtest %*% beta
#' }
#' cv <- cv.setup(x, y, score=mean_se, num_folds = 5, num_iter = 2)
#' res <- cv.particle_swarm(cv, regr, logreg = c(-3, 5), num_particles = 3, num_generations = 4, maximize = FALSE)
cv.particle_swarm <- function(setup, f, ..., num_particles=5, num_generations=10, maximize = TRUE, nested = FALSE) {
  if (nested) stop("Nested cross-validation is not yet supported.")
  args <- list(...)
  if (length(args) == 0)
    stop("Please provide parameter intervals for f, like cv.particle_swarm(setup, f, param1=c(0, 20)).")
  check_cv_args(f, args)
  
  fcv <- function(...) cv.run(setup, f, ...)$score.mean
  
  res <- particle_swarm(fcv, ..., 
                        num_particles=num_particles, 
                        num_generations=num_generations, 
                        maximize = maximize)
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

#' Scoring functions for measuring model performance
#' @param ytrue  true (test) values for y
#' @param yhat   estimated values for y
#' @param yprob  estimated probabilities of positive class, used for binary classification
#' @param yscore ranking scores for computing AUC for ROC and Precision-Recall curve
#' @param decreasing whether high score corresponds to TRUE (positive) class or FALSE (negative) class
#' @param top    value between 0.0 and 1.0, the proportion of top ratings used, if 1.0 all are used for computing AUC, if 0.1 then AUC only depends on top 10\%
#' @param alpha  how many of the top ranked samples are important, used in early discovery metrics to define what is considered 'early', default 20.
#' @name  Scoring
NULL

#' @rdname Scoring
#' @return mean of squared errors between ytrue and yhat
#' @export
mean_se <- function(ytrue, yhat) {
  mean((ytrue-yhat)^2)
}

#' @rdname Scoring
#' @return mean of absolute errors between ytrue and yhat
#' @export
mean_ae <- function(ytrue, yhat) {
  mean(abs(ytrue-yhat))
}

#' @rdname Scoring
#' @return proportion of correct predictions with yprob > 0.5 is used as cut-off
#' @export
accuracy_from_p <- function(ytrue, yprob) {
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

#' @rdname Scoring
#' @return proportion of correct predictions, i.e. \code{mean(ytrue==yhat)}
#' @export
accuracy <- function(ytrue, yhat) {
  mean(ytrue==yhat)
}

#' @rdname Scoring
#' @return Area under ROC curve
#' @export
auc_roc <- function(ytrue, yscore, decreasing=TRUE, top=1.0) {
  if (length(ytrue) != length(yscore)) {
    stop(sprintf("length of ytrue(%d) should be the same as length of ycore(%d).", 
                 length(ytrue), length(yscore) ))
  }
  tryCatch( {
    pred <- ROCR::prediction(yscore,  ytrue)
    return( performance(pred, "auc")@y.values[[1]] )
  }, error = function(e) return(NA_real_)
  )
}

#' @rdname Scoring
#' @return Area under precision-recall curve
#' @export
auc_pr <- function(ytrue, yscore, decreasing=TRUE) {
  tryCatch( {
    perf <- ROCR::performance(ROCR::prediction(yscore, ytrue), "prec", "rec")
    prec <- perf@y.values[[1]]
    if (is.na(prec[1]))
      prec[1] <- 1.0
    rec  <- perf@x.values[[1]]
    aver <- (prec[-1] + prec[-length(prec)]) / 2
    return( sum(base::diff(rec) * aver) )
  }, error = function(e) return(NA_real_))
}

to1or0 <- function(y) {
  if (is.factor(y)) {
    if (length(levels(y)) > 2) stop("Expected two-level factor, got more.")
    return( (0:1)[as.numeric(y)] )
  }
  if (max(y) == 2) return(y-1)
  return(y)
}

#' @rdname Scoring
#' @return Early discovery metric RIE (robust initial enhancement)
#' @export
early_rie <- function(ytrue, yscore, decreasing=TRUE, alpha=20.0) {
  enrichvs::rie(yscore, to1or0(ytrue), decreasing=decreasing, alpha=alpha)
}

#' @rdname Scoring
#' @return Early discovery metric BEDROC (Boltzmann-enhanced discrimination of ROC)
#' @export
early_bedroc <- function(ytrue, yscore, decreasing=TRUE, alpha=20.0) {
  enrichvs::bedroc(yscore, to1or0(ytrue), decreasing=TRUE, alpha=20.0)
}


