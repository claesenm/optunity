
py2r_name = "/tmp/py2r_pipe"

debug <- function() {
  getOption("optunity-debug", default=FALSE)
}

readpipe <- function(pipe){
  reply <- readLines(pipe, n=1)
  if (length(reply)==0) stop('Optunity error: broken pipe.')
  return (reply)
}

send <- function(r2py, data){
  if (debug()) {
    print(paste("[Optunity] to python:", toJSON(data) ))
  }
  cat(toJSON(data), '\n', file=r2py)
  flush(r2py)
}

receive <- function(py2r){
  message <- readpipe(py2r)
  reply <- fromJSON(message)
  if ("error_msg" %in% names(reply)){
    stop(paste("Optunity error: ", reply$error_msg))
  }
  if (debug()) {
    print(paste("[Optunity] from python:", message))
  }
  return (reply)
}

launch <- function(){
  # http://stackoverflow.com/a/5561188/2148672
  for (i in 1:10) {
    rnd <- sprintf("%x", sample.int(n = 10000000L, size=1))
    py2r_name_rand <- sprintf("%s-%s", py2r_name, rnd)
    if ( ! file.exists(py2r_name_rand))
      break
  }
  
  system(paste('rm -f',py2r_name_rand, sep=' '))
  system(paste('mkfifo',py2r_name_rand, sep=" "))
  optunitydir <- find.package("optunity")
  cmd <- sprintf("cd '%s'; python -m optunity.piped > '%s' 2>/dev/null",
                 optunitydir, 
                 py2r_name_rand)
  r2py <- pipe(cmd, 'w')
  py2r <- fifo(py2r_name_rand,'r', blocking=TRUE)
  conn <- list(py2r = py2r, r2py = r2py, py2r_name=py2r_name_rand)
  return (conn)
}

close_pipes <- function(cons){
  close(cons$py2r)
  close(cons$r2py)
  system(paste('rm -f ',cons$py2r_name,sep=''))
}
