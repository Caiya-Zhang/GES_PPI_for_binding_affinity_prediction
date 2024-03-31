# GES_PPI

## Description
This repo contains the code and data for our paper "An Interpretable Deep Geometric Learning Model to Predict the Effects of Mutations on Protein-Protein Interactions Using Large-scale Protein Language Model". Here we developed a geometric deep learning model designed to predict changes in binding affinity for protein-ligand interactions. The model employs a graph Transformer framework that can learn long-distance relationships, embeds a gated graph neural network, and incorporates a pre-trained large-scale protein language model to ultimately achieve the prediction of ΔΔG. The efficacy of the model is thoroughly evaluated experimentally and a reasonable interpretability analysis is given. This new approach to study protein stability alterations by implementing geometric deep learning methods has important implications for future stability prediction efforts.


## Dependency
* Python 3.7
* CUDA
* Pytorch
* Rosetta


## Installation
GES_PPI's ΔΔG prediction involves two main steps as shown below:
(You may need to install conda in advance.)

### 1. Install Rosetta 3
a. Go to https://els2.comotion.uw.edu/product/rosetta to get an academic license for Rosetta.
b. Download Rosetta 3.13 (source + binaries for Linux) from this site: https://www.rosettacommons.org/software/license-and-download
c. Extract the tarball to a local directory from which Rosetta binaries can be called by specifying their full path.

### 2. Clone GES_PPI by

```
https://github.com/Caiya-Zhang/GES_PPI_for_binding_affinity_prediction.git
```



# Run GES_PPI

1. All the required environments are in requirement.yml.
```
conda env create -f requirement.yml
```

2. To run the experiments, please refer to the commands below (taking OGBG-Code2 as an example):
```
python main.py --configs configs/code2/gnn-transformer/no-virtual/run_2.yml --runs 5
```