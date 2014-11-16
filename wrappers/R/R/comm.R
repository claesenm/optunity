
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
  optunitydir <- dirname( system.file("optunity", package="optunity") )
  cmd <- sprintf("cd '%s'; python -m optunity.standalone server %s", optunitydir, ifelse(debug(), "", "2>/dev/null"))

  opipe <- pipe(cmd, open="r+")
  portstr <- readLines(opipe, n = 1)
  port    <- strtoi( gsub("^\\s+|\\s+$", "", portstr) )

  if (is.na(port) || port == 0) {
    stop("Optunity error: could not launch python process.")
  }

  socket <- socketConnection(port = port, blocking = TRUE)
  conn   <- list(socket = socket, port = port, opipe = opipe)
  return (conn)
}

close_pipes <- function(cons){
  close(cons$socket)
  close(cons$opipe)
}
