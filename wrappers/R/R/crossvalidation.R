
cv.setup <- function(x, y=NULL, num_folds=5, num_iter=1, 
                     strata=NULL, clusters=NULL,
                     seed=NULL) {
  if ( ! is.matrix(x) && ! is.data.frame(x))
    stop("x has to be either a matrix or data.frame")
  
  setup <- list()
  setup$supervised = ! is.null(y)
  if (setup$supervised && nrow(x) != length(y))
    stop( sprintf("Number of rows of x is not equal to the length of y.", nrow(x), length(y)) )
  setup$x = x
  setup$y = y
  setup$num_folds = num_folds
  setup$seed = seed
  if ( ! is.null(seed) ) set.seed(seed)
  
  setup$folds = generate_folds(nrow(x), num_folds=num_folds, num_iter=num_iter,
                               strata=strata, clusters=clusters)
  class(setup) <- "cv.setup"
  return(setup)
}
