Installation from git
--------------------

Install necessary dependencies in R (if they are not yet installed)
```R
install.packages(c("rjson", "ROCR", "enrichvs", "plyr"))
```
and then do (in shell)
```bash
git clone https://github.com/claesenm/optunity
cd optunity/wrappers
R CMD build R/
R CMD INSTALL optunity_*.tar.gz
```


