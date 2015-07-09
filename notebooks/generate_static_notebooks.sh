#!/bin/bash
# this script requires ipython and pandoc/nodejs

for i in $(ls *.ipynb);
do
    echo "Processing $i."
    ipython nbconvert $i;
done
