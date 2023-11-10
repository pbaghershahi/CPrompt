#!/bin/bash

pytorch_version=$(python -c "import torch; print(f'torch-{torch.__version__}.html')")
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/$pytorch_version
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/$pytorch_version
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/$pytorch_version
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/$pytorch_version
pip install torch-geometric
pip install osmnx
python -c "from IPython.display import clear_output; clear_output(); print('Installed requirements')"
