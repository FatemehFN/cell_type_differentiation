
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

datasetName = "ROSMAP24"
variantName = "complete_complete"

dataRawPath = Path("Data/Raw")
dataPreprocessedPath = Path("Data/Preprocessed")

dataInputPath = dataRawPath / datasetName


# reducedNetworkCellTypes = set(['CD56+ NK', 'CD19+ B', 'CD4+/CD25 T Reg'])

expressionDataPath = dataInputPath/("ROSMAP_24_exp_matrix.csv")
geneAnnotationPath = dataInputPath/("cell_types_ROSMAP_24.txt")
cellTypesDataPath = dataInputPath/("cell_types_ROSMAP_24.txt")

# format each line have: "cellType : cellName"
cell2Type = {}
with open(cellTypesDataPath) as f:
    for line in f:
        line = line.strip()
        cellType, cellName = line.split(" : ")
        cell2Type[cellName.strip()] = cellType.strip()

# column 0 is geneNames, but it need to be named there
expressionData = pd.read_csv(expressionDataPath)

expressionMatrix = expressionData.iloc[:,1:]
expressionMatrix = expressionMatrix.to_numpy(dtype="float64").T

#expressionMatrixShape = (cells,genes)
geneNames = expressionData.iloc[:,0].to_numpy()
cellsNames = expressionData.columns[1:].str.replace(".","-").to_numpy()


allCellsWithCellType = list(cell2Type.keys())
# write both to txt files ROSMAP_cells.txt and ROSMAP_genes.txt
# write allCells with cell type ROSMAP_allcells.txt
cellTypes = np.array([cell2Type[cell]  if cell in cell2Type else None for cell in cellsNames] )

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

