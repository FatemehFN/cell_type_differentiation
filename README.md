# Cell Type Differentiation Using Network Clustering Algorithms

This repository contains code and analysis for the paper _"Cell Type Differentiation Using Network Clustering Algorithms"_, which compares the performance of several graph-based clustering methods in identifying cell types from single-cell RNA sequencing (scRNA-seq) data.

## 📁 Repository Structure


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
