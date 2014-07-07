library("rjson")

# TODO: fix me
py2r_name = "/data/svn/claesenm/python/pipes/py2r_pipe"

readpipe <- function(pipe){
    reply <- readLines(pipe)
    if (length(reply)==0) stop('Optunity error: broken pipe.')
    return (reply)
}

send <- function(r2py, data){
    cat(toJSON(data), '\n', file=r2py)
    flush(r2py)
}

receive <- function(py2r){
    reply <- fromJSON(readpipe(py2r))
    if ("error_msg" %in% names(reply)){
        stop(paste("Optunity error: ", reply$error_msg))
    }
    return (reply)
}

launch <- function(){
    # http://stackoverflow.com/a/5561188/2148672
    # FIXME: fifo does not exist on windows
    system(paste('rm -f',py2r_name,sep=' '))
    system(paste('mkfifo',py2r_name,sep=" "))
    cmd <- paste('python -m optunity.piped >',py2r_name,sep=' ')
    r2py <- pipe(cmd, 'w')
    py2r <- fifo(py2r_name,'r', blocking=TRUE)
    conn <- list(py2r = py2r, r2py = r2py)
    return (conn)
}

close_pipes <- function(r2py, py2r){
    close(py2r)
    close(r2py)
    system(paste('rm -f ',py2r_name,sep=''))
}
