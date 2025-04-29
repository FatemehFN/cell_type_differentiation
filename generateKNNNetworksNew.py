
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


enablePCA = True
shouldNormalize = True
sampleRate = 1.0
kneighbors = 30


datasetName = "ROSMAP24"
variantName = "complete_complete"

dataPreprocessedPath = Path("Data/Preprocessed")

dataFilePath = dataPreprocessedPath/(datasetName+"_"+variantName+"_data.npz")

networksPath = Path("Networks")/(datasetName+"_"+variantName)
networksPath.mkdir(parents=True,exist_ok=True)

data = np.load(dataFilePath,allow_pickle=True)
aggregatedMatrix = data["aggregatedMatrix"]
geneIDs = data["geneIDs"]
cellIDs = data["cellIDs"]
cellTypes = data["cellTypes"]


enablePCA = True
shouldNormalize = True
kneighbors=30

sampleRate = 1.0

if(sampleRate<1.0):
    sampleIndices = np.random.choice(aggregatedMatrix.shape[0],int(aggregatedMatrix.shape[0]*sampleRate),replace=False)
else:
    sampleIndices = np.arange(aggregatedMatrix.shape[0])

sampleIndices = sampleIndices[np.sum(aggregatedMatrix[sampleIndices,:],axis=1)>=100]


for useFeatureSelection in [True, False]:
    if(useFeatureSelection):
        adata = ad.AnnData(aggregatedMatrix[sampleIndices,:].astype(np.float64))
        sc.pp.highly_variable_genes(adata, flavor = "seurat_v3", n_top_genes=2000)
        variableGeneIDs = geneIDs[adata.var.highly_variable]
        adata = adata[:, adata.var.highly_variable]
        featuresMatrix = adata.X
    else:
        featuresMatrix = aggregatedMatrix[sampleIndices,:].astype(np.float64)

    for useLogNormalization in [False, True]:
        for shouldNormalize in [False, True]:
            if(useLogNormalization):
                normalizedMatrix = np.log1p(featuresMatrix)
            else:
                normalizedMatrix = featuresMatrix.copy()

            if(shouldNormalize):
                for i in tqdm(range(normalizedMatrix.shape[0]),desc="Normalizing matrix"):
                    # add random noise to each column Make sure they are degenerate
                    normalizedMatrix[i,:] += np.random.normal(0,0.01,normalizedMatrix.shape[1])

            for enablePCA in [False, True]:
                if enablePCA:
                    pca = sklearn.decomposition.PCA(n_components=50)
                    # reduce columns by applying PCA
                    # First standardize columns
                    # print("Standardizing columns")
                    # normalizedMatrix = sklearn.preprocessing.scale(normalizedMatrix,axis=0)
                    # This is not working because of memory error. normalize it in place and chunk by chunk
                    pcaMatrix = normalizedMatrix.copy()
                    averageVector = np.mean(pcaMatrix,axis=0)
                    stdVector = np.std(pcaMatrix,axis=0)
                    for i in tqdm(range(pcaMatrix.shape[0]),desc="Standardizing matrix"):
                        pcaMatrix[i,:] = pcaMatrix[i,:] - averageVector
                        pcaMatrix[i,:] = pcaMatrix[i,:] / stdVector
                    # make nan 0
                    pcaMatrix[np.isnan(pcaMatrix)] = 0

                    pca.fit(pcaMatrix)
                    networkMatrix = pca.transform(pcaMatrix)
                else:
                    networkMatrix = normalizedMatrix

                networkMatrix = networkMatrix.copy()
                # inplace log(x+1) transformation
                # remove mean inplace
                # divide by standard deviation inplace
                for i in tqdm(range(networkMatrix.shape[0]),desc="Normalizing matrix"):
                    networkMatrix[i,:] = networkMatrix[i,:] - networkMatrix[i,:].mean()
                    networkMatrix[i,:] = networkMatrix[i,:] / networkMatrix[i,:].std()
                    # Also try normalization by the L2 norm

                # # remove 0 entries from sampleIndices
                # normalizedMatrix = normalizedMatrix[np.sum(normalizedMatrix,axis=1)!=0,:]
                cellIDsSampled = cellIDs[sampleIndices]



                # knnData = nearest_neighbors(normalizedMatrix,
                #                             n_neighbors=20,
                #                             metric="correlation",
                #                             metric_kwds=None,
                #                             angular=False,
                #                             random_state=None,
                #                             verbose=True
                #                             )

                for kneighbors in [4,5,6,7,8,9,10,20,30]:
                    variant = ""
                    if enablePCA:
                        variant = "_PCA"
                    if shouldNormalize:
                        variant += "_normalized"
                    if sampleRate<1.0:
                        variant += f"_sample{sampleRate}"
                    if useLogNormalization:
                        variant += "_log"
                    if useFeatureSelection:
                        variant += "_featureSelection"
                    if(networksPath/f"knn_{kneighbors}_{variant}_pos.xnet").exists():
                        print(f"skipping knn_{kneighbors}_{variant}_pos.xnet")
                        continue
                    # try 4 times with no exception
                    trials =4
                    while trials>0:
                        try:
                            knnData = NNDescent(networkMatrix,
                                                metric="correlation",
                                                n_neighbors=kneighbors,
                                                verbose=True
                            ).neighbor_graph
                            break
                        except Exception as e:
                            print(e)
                            trials -= 1
                            if trials==0:
                                break
                    if(trials==0):
                        print(f"Failed to compute knn for {kneighbors} neighbors")
                        continue
                    edges = []
                    weights = []
                    neighborIndices = []
                    for sourceIndex,theNeighbors in enumerate(tqdm(knnData[0])):
                        for neighborIndex,neighbor in enumerate(theNeighbors):
                            if(sourceIndex==neighbor):
                                continue
                            similarity = np.mean(networkMatrix[sourceIndex,:]*networkMatrix[neighbor,:])
                            edges.append((sourceIndex,neighbor))
                            weights.append(similarity)
                            neighborIndices.append(neighborIndex)
                        # edges.append((kneighbors[0],kneighbors[1]))
                        # edges.append((kneighbors[0],kneighbors[2]))


                    g = ig.Graph(networkMatrix.shape[0],edges=edges,edge_attrs={"weight":weights, "neighborIndex":neighborIndices})
                    # remove any self loops
                    g = g.simplify(multiple=False,loops=True,combine_edges={"weight":"sum"})

                    g.vs["Label"] = cellIDsSampled
                    g.vs["CellType"] = np.array(cellTypes)[sampleIndices]



                    xn.igraph2xnet(g,(networksPath/f"knn_{kneighbors}_{variant}.xnet"))
                    #remove links with 0 or negative weights
                    g.es.select(weight_lt=0).delete()
                    xn.igraph2xnet(g,(networksPath/f"knn_{kneighbors}_{variant}_pos.xnet"))


                    # save gml
                    # g.write_gml(str((/("ROSMAP-465_network_sample%.3f.gml"%sampleRate)).absolute()))
                    # aggregatedMatrixSampled = aggregatedMatrix[sampleIndices,:]
                    # # save csv of node attributes + whole data from the sample
                    # # Also include columns for Label, Index and CellType, SubCellType
                    # nodeAttributes = pd.DataFrame({"Label":g.vs["Label"],
                    #                                     "Index":g.vs["Index"],
                    #                                     "CellType":g.vs["CellType"],
                    #                                     "SubCellType":g.vs["SubCellType"]})
                    # # include gene values from aggregatedMatrixSampled
                    # # rename columns names to geneIDs for the agregatedMatrixSampled repreentation
                    # aggregatedMatrixSampled = pd.DataFrame(aggregatedMatrixSampled,columns=geneIDs)
                    # nodeAttributes = pd.concat([nodeAttributes,aggregatedMatrixSampled],axis=1)
                    # nodeAttributes.to_csv(samplesPath/("ROSMAP-465_network_sample%.3f_nodeAttributes.csv"%sampleRate),index=False)
