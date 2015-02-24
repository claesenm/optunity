library("optunity")

## fold generation
strata <- list(c(1,2,3), c(6,7,8,9))
folds <- generate_folds(20, num_folds=10, num_iter=2, strata=strata)


## solver manual request
manual()

## solver demo
f <- function(x, y) return (-x^2-y^2)
cfg <- list(x = seq(-5, 5, by=0.5), y = seq(-5, 5, by=0.5))
result <- optimize2(f, solver_name='grid search', solver_config=cfg, call_log=TRUE)

## particle swarms
f   <- function(x,y) -x^2 - y^2
opt <- particle_swarm(f, x=c(-5, 5), y=c(-5, 5) )

## cv of ridge regression
## artificial data
N <- 50
x <- matrix(runif(N*5), N, 5)
y <- x[,1] + 0.5*x[,2] + 0.1*runif(N)

## ridge regression
regr <- function(x, y, xtest, ytest, logC) {
     ## regularization matrix
     C =  diag(x=exp(logC), ncol(x))
   beta = solve(t(x) %*% x + C, t(x) %*% y)
      xtest %*% beta
}
cv <- cv.setup(x, y, score=mean_se, num_folds = 5, num_iter = 2)
res <- cv.particle_swarm(cv, regr, logC = c(-5, 5), maximize = FALSE)

## optimal value for logC:
res$solution$logC
