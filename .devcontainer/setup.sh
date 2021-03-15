#!/bin/bash
if [ ! -d ".git" ]; then
    git clone https://github.com/fromm1990/GomapClustering.git .
    pip install -e ./src/
else
    echo "Repo already cloned"
fi