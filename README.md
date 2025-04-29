# Cell Type Differentiation Using Network Clustering Algorithms

This repository contains code and analysis for the paper _"Cell Type Differentiation Using Network Clustering Algorithms"_, which compares the performance of several graph-based clustering methods in identifying cell types from single-cell RNA sequencing (scRNA-seq) data.

## 📁 Repository Content
```bash
scVI_pbmc68k.ipynb # Benchmarking cell type clustering with scVI 
preprocessROSMAP24_complete.py # Preprocessing ROSMAP24 and generating KNN graphs 
generateKNNNetworksNew.py # KNN graph construction using NNDescent 
getcommunities.py # Runs Leiden, Infomap, and SBM on graphs 
pbmc58k_infomap_time.py # Infomap timing and clustering evaluation 
pbmc68k_nested_sbm_time.py # Nested SBM clustering evaluation
pbmc68k_nested_leiden_time.py # Leiden clustering evaluation
pbmc68_visualization.py # Visualization of clustering results
```
---


## 🧬 Project Overview

We apply multiple community detection algorithms on similarity graphs derived from scRNA-seq data to benchmark how well they recover known cell types. This framework allows systematic comparison of modularity-based and probabilistic clustering approaches, including their runtime efficiency.

---

## ⚙️ Setup

### Requirements

- Python 3.8+
- Core dependencies:
  - `scanpy`, `scvi-tools`, `igraph`, `graph-tool`, `leidenalg`, `infomap`, `xnetwork`, `pynndescent`
  - `scikit-learn`, `umap-learn`, `pandas`, `numpy`, `matplotlib`, `clusim`, `tqdm`, `oslom`

Install most dependencies with:

```bash
pip install scanpy scvi-tools igraph leidenalg infomap xnetwork pynndescent scikit-learn umap-learn tqdm pandas numpy matplotlib
```
`graph-tool` requires manual installation.

## 🔁 Wrokflow

### 1. Preprocess Expression Data
```bash
python preprocessROSMAP24_complete.py
```
### 2. Generate KNN Graphs
```bash
python generateKNNNetworksNew.py
```
### 3. Apply Clustering Methods
```bash
python getcommunities.py
```
### 4. Benchmark Runtime and Accuracy
```bash
python pbmc58k_infomap_time.py
python pbmc68k_nested_sbm_time.py
```
### 5. Visualize Networks and Clustering Results
```bash
python pbmc68_visualization.py
```
### 6. Run scVI Embedding

Open and execute the notebook 
```bash
scVI_pbmc68k.ipynb
```



