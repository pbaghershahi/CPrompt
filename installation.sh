#!/bin/bash

curr_version=$(python -c "import torch; print(torch.__version__)")
pytorch_version="${1:-$curr_version}"
pytorch_version="torch-${pytorch_version}.html"
pip install --no-index pyg_lib==0.3.1 torch-scatter torch-sparse \
torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/$pytorch_version
pip install torch-geometric
printf "\033c"