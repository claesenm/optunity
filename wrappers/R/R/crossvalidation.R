
cv.setup <- function(x, y=NULL, score, num_folds=5, num_iter=1, 
                     strata=NULL, clusters=NULL,
                     seed=NULL) {
  if ( ! is.matrix(x) && ! is.data.frame(x))
    stop("x has to be either a matrix or data.frame")
  
  if (missing(score))
    stop("Please set score to NULL or provide score function(ytrue, yscore), like score.accuracy for classification or score.neg.mse for regression. score=NULL means f returns score instead of predictions.")
  
  setup <- list()
  setup$x = x
  setup$y = y
  setup$score = score
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
  
  scores = sapply(1:setup$num_iter, function(iter) {
    sapply(1:setup$num_folds, function(fold) {
      itrain = setup$folds[,iter] != fold
      itest  = ! itrain
      xtrain = setup$x[ itrain, ]
      xtest  = setup$x[ itest,  ]
      ytrain = setup$y[ itrain ]
      ytest  = setup$y[ itest  ]
      yhat <- f(xtrain, ytrain, xtest, ytest, ...)
      if (is.null(setup$score)) {
        s <- yhat
        if (length(s) > 1) stop("f returned a vector, but should return 1 numeric value (score).")
        if ( ! is.numeric(s)) stop("f returned non-numeric value.")
      } else {
        s <- setup$score(ytest, yhat)
      }
      return(s)
    })
  })
  out <- list()
  out$scores     = scores
  out$score.mean = mean(scores)
  out$score.sd   = sd(scores)
  out$score.iter.mean = colMeans(scores)
  return( out )
}

score.neg.mse <- function(ytrue, yhat) {
  - mean((ytrue-yhat)^2)
}

score.accuracy <- function(ytrue, yhat) {
  mean(ytrue==yhat)
}

score.roc.auc <- function(ytrue, yscore, decreasing=TRUE, top=1.0) {
  enrichvs::auc(yscore, ytrue, decreasing=decreasing, top=top)
}

score.pr.auc <- function(ytrue, yscore, decreasing=TRUE) {
  perf <- ROCR::performance(ROCR::prediction(yscore, ytrue), "prec", "rec")
  prec <- perf@y.values[[1]]
  if (is.na(prec[1]))
    prec[1] <- 1.0
  rec  <- perf@x.values[[1]]
  aver <- (prec[-1] + prec[-length(prec)]) / 2
  return( sum(base::diff(rec) * aver) )
}

score.rie <- function(ytrue, yscore, decreasing=TRUE, alpha=20.0) {
  enrichvs::rie(yscore, ytrue, decreasing=decreasing, alpha=alpha)
}

score.bedroc <- function(ytrue, yscore, decreasing=TRUE, alpha=20.0) {
  enrichvs::bedroc(yscore, ytrue, decreasing=TRUE, alpha=20.0)
}


