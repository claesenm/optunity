library("rjson")

# TODO: fix me
py2r_name = "/data/svn/claesenm/python/pipes/py2r_pipe"

readpipe <- function(pipe){
    return (readLines(pipe))
}

send <- function(r2py, data){
    cat(toJSON(data), '\n', file=r2py)
    flush(r2py)
}

receive <- function(py2r){
    return (fromJSON(readpipe(py2r)))
}

# http://stackoverflow.com/a/5561188/2148672
system(paste('rm -f',py2r_name,sep=' '))
system(paste('mkfifo',py2r_name,sep=" "))
cmd <- paste('python -m optunity.piped >',py2r_name,sep=' ')

r2py <- pipe(cmd, 'w')
py2r <- fifo(py2r_name,'r', blocking=TRUE)

msg <- c()
msg$manual <- TRUE

# send message
send(r2py, msg)

# receive message
content <- receive(py2r)
cat(content$manual, sep="\n")

# clean up
close(py2r)
close(r2py)
system(paste('rm -f ',py2r_name,sep=''))
