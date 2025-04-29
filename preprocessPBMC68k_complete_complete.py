
import numpy as np
from umap.umap_ import nearest_neighbors
from pathlib import Path
from tqdm.auto import tqdm
import sklearn
import pandas as pd
import igraph as ig
import xnetwork as xn
from pynndescent import NNDescent
import igraph as ig
# Path: Scripts/analysis.py
import pandas as pd
import numpy as np
import scipy
import xnetwork as xn
import pickle
from tqdm.auto import tqdm
from pathlib import Path
from mpmath import mp # pip install mpmath
from scipy import integrate
import scanpy as sc
import anndata as ad

datasetName = "pbmc68k"
variantName = "complete_complete"

dataRawPath = Path("Data/Raw")
dataPreprocessedPath = Path("Data/Preprocessed")

dataInputPath = dataRawPath / datasetName


# reducedNetworkCellTypes = set(['CD56+ NK', 'CD19+ B', 'CD4+/CD25 T Reg'])

expressionMatrixPath = dataInputPath/("pbmc68k_expression_matrix.mtx")
geneNamesPath = dataInputPath/("genes.txt")
cellsNamesPath = dataInputPath/("cell barcodes.txt")
geneAnnotationPath = dataInputPath/("68k_pbmc_barcodes_annotation.tsv")

expressionMatrix = scipy.io.mmread(expressionMatrixPath)
expressionMatrix = expressionMatrix.toarray()

#expressionMatrixShape = (cells,genes)
geneNames = np.genfromtxt(geneNamesPath, dtype=str) #
cellsNames = np.genfromtxt(cellsNamesPath, dtype=str) #
geneAnnotation = pd.read_csv(geneAnnotationPath, sep="\t") # 
# TSNE.1	TSNE.2	barcodes	celltype
barcode2celltype = dict(zip(geneAnnotation.barcodes, geneAnnotation.celltype))
cellTypes = np.array([barcode2celltype[cell] for cell in cellsNames])

# remove 0 columns and rows (also remove from respective names and cellTypes)
geneNames = geneNames[~np.all(expressionMatrix == 0, axis=0)]
cellsNames = cellsNames[~np.all(expressionMatrix == 0, axis=1)]
cellTypes = cellTypes[~np.all(expressionMatrix == 0, axis=1)]
expressionMatrix = expressionMatrix[~np.all(expressionMatrix == 0, axis=1)]
expressionMatrix = expressionMatrix[:, ~np.all(expressionMatrix == 0, axis=0)]


aggregatedMatrix = expressionMatrix

geneIDs = geneNames
cellIDs = cellsNames

# save geneIDs, cellIDs, aggregatedMatrix, cellTypes npx
np.savez(dataPreprocessedPath/(datasetName+"_"+variantName+"_data.npz"), geneIDs=geneIDs, cellIDs=cellIDs, aggregatedMatrix=aggregatedMatrix, cellTypes=cellTypes)

