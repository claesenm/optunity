Installation from git
--------------------

Make sure you have necessary R packages installed (rjson) and then do
```bash
git clone https://github.com/claesenm/optunity
cd optunity/wrappers
R CMD build R/
R CMD INSTALL optunity_*.tar.gz
```


