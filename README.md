# Tree-Transformer

Code for paper "Learning Program Representations with a Tree-Structured Transformer" (accepted by SANER 2023).

## Requirements

PyTorch>=1.5

dgl>=0.5

torchtext==0.6

pycparser

tree-sitter (with Python grammar)

anytree

## Data preprocessing

POJ: python preprocess_poj.py

Wrong operator localization and repair: python preprocess_loc.py

CodeNet dataset do not require separate preprocessing.

## Running

POJ: python run_poj.py

CodeNet: python run_codenet.py

Wrong operator localization and repair: python run_loc.py

