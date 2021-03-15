#!/bin/bash
if [! -d ".git"] then
    echo "Repo already cloned"
else
    git clone https://github.com/fromm1990/GomapClustering.git .
    pip install -e ./src/
fi