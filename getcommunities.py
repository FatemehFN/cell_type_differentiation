
import clusim.sim as csim
import leidenalg
from igraph import Graph
from clusim.clustering import Clustering, print_clustering
import pandas as pd
from infomap import Infomap
import numpy as np
import xnetwork as xn
import graph_tool as gt
import graph_tool.inference as gtInference
from graph_tool import Graph as gtGraph
from pathlib import Path 
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
import oslom.oslom as oslom

        # fromCluster = Clustering()
        # fromCluster.from_membership_list(partition_ROSMAP.membership)
        # toCluster = Clustering()
        # toCluster.from_membership_list(cell_type_membership_vector)


        # ariValue = csim.adjrand_index(fromCluster,toCluster)


datasetName = "ROSMAP24all"
variantName = "complete_complete"

dataPreprocessedPath = Path("Data/Preprocessed")

dataFilePath = dataPreprocessedPath/(datasetName+"_"+variantName+"_data.npz")

networksPath = Path("Networks")/(datasetName+"_"+variantName)
seuratNetworksPath = Path("SeuratNetworks")
networksWithCommunitiesPath = Path("NetworksWithCommunities")/(datasetName+"_"+variantName)
networksWithCommunitiesPath.mkdir(parents=True,exist_ok=True)




def leiden(ROSMAP_graph,weights,resolution=1.0):
    best_q=None
    bestPartition = None
    for i in range(20):
        partition_ROSMAP = leidenalg.find_partition(ROSMAP_graph,leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution,weights=weights)
        
        q=partition_ROSMAP.quality()
        if best_q is None or q>best_q:
            bestPartition = partition_ROSMAP
            best_q=q
    
    return [str(entry) for entry in bestPartition.membership]





def infomapApply(g, m_time, weights=None):
    vertexCount = g.vcount()
    if(weights):
        edges = [(e.source, e.target, e[weights]) for e in g.es]
    else:
        edges = g.get_edgelist()

    if(g.is_directed()):
        extraOptions = "-d"
    else:
        extraOptions = ""
    im = Infomap("%s -N 10 --silent --seed %d" %
                (extraOptions, np.random.randint(4294967296)),markov_time=m_time)
    
    im.setVerbosity(0)
    for nodeIndex in range(0, vertexCount):
        im.add_node(nodeIndex)
    for edge in edges:
        if(len(edge) > 2):
            if(edge[2]>0):
                im.addLink(edge[0], edge[1], edge[2])
            im.add_link(edge[0], edge[1], weight=edge[2])
        else:
            im.add_link(edge[0], edge[1])

    im.run()
    membership = [":".join([str(a) for a in membership])
                for index, membership in im.get_multilevel_modules().items()]

    levelMembership = []

    #print(max([len(element.split(":")) for element in membership]))
    levelCount = max([len(element.split(":")) for element in membership])
    for level in range(levelCount):
        levelMembership.append(
            [":".join(element.split(":")[:(level+1)]) for element in membership]
        )
    return levelMembership



def SBMMinimizeMembershipNested(graph,weights= None, weightMode="real-normal"):
    vertexCount = graph.vcount()

    g = gt.Graph(directed=graph.is_directed())
    for _ in range(vertexCount):
        g.add_vertex()

    weighted = bool(weights)
    if(weights):
        weightsProperty = g.new_edge_property("double")

    for edge in graph.es:
        gedge = g.add_edge(edge.source, edge.target)
        if(weighted):
            weight = edge[weights]
            weightsProperty[gedge] = weight
    if(weighted):
        g.edge_properties[weights] = weightsProperty
    
    state_args = {}
    if(weighted):
        state_args["recs"] = [g.ep.weight]
        state_args["rec_types"] = [weightMode]

    state_args["deg_corr"] = True
    # print(state_args)
    state = gtInference.minimize.minimize_nested_blockmodel_dl(
        g, state_args=state_args)
    levelMembership = []
    levels = state.get_levels()
    lastIndices = list(range(vertexCount))
    for level, s in enumerate(levels):
        blocks = s.get_blocks()
        lastIndices = [blocks[gindex] for gindex in lastIndices]
        if(len(set(lastIndices)) == 1 and level != 0):
            break
        levelMembership.append([str(entry) for entry in lastIndices])
    levelMembership.reverse()
    return levelMembership


def SBMMinimizeMembership(g, degreeCorrected=True,**kwargs):
    vertexCount = g.vcount()
    gGT = gtGraph(directed=g.is_directed())
    for _ in range(0, vertexCount):
        gGT.add_vertex()
    for edge in g.get_edgelist():
        gGT.add_edge(edge[0], edge[1])

    state = gtInference.minimize.minimize_blockmodel_dl(
        gGT, state_args={"deg_corr": degreeCorrected})
    DLDetected = state.entropy(**kwargs)
    DLTrivial = gtInference.blockmodel.BlockState(
        gGT, B=1, deg_corr=degreeCorrected).entropy(**kwargs)
    return [str(entry) for entry in list(state.get_blocks())]


def SBMMinimizeMembershipWeighted(graph, weightMode="real-normal", degreeCorrected=True):
    vertexCount = graph.vcount()

    g = gt.Graph(directed=graph.is_directed())
    for _ in range(vertexCount):
        g.add_vertex()

    weighted = "weight" in graph.edge_attributes()
    if(weighted):
        weightsProperty = g.new_edge_property("double")

    for edge in graph.es:
        gedge = g.add_edge(edge.source, edge.target)
        if(weighted):
            weight = edge["weight"]
            weightsProperty[gedge] = weight
    if(weighted):
        g.edge_properties["weight"] = weightsProperty


    
    state_args = {}
    if(weighted):
        state_args["recs"] = [g.ep.weight]
        state_args["rec_types"] = [weightMode]

    state_args["deg_corr"] = degreeCorrected
    # print(state_args)

    state = gtInference.minimize.minimize_blockmodel_dl(
        g, state_args=state_args)

    DLTrivial = gtInference.blockmodel.BlockState(
        g, B=1, **state_args).entropy()
    
    DLDetected = state.entropy()
    return [str(entry) for entry in list(state.get_blocks())]

resolutions = [0.001,0.005,0.01,0.1,1.0,0.5,5.0,10.0,20.0,50,100]
markovTimes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5,10]


def calculateMetrics(g,propertyName):
    fromCluster = Clustering()
    fromCluster.from_membership_list(g.vs["CellType"])
    toCluster = Clustering()
    toCluster.from_membership_list(g.vs[propertyName])
    adjRandIndex = csim.adjrand_index(fromCluster,toCluster)
    adjMutualInfo = csim.adj_mi(fromCluster,toCluster)
    return {
        "ARI": adjRandIndex,
        "AMI": adjMutualInfo
        }

def processnetwork(networkFile):
    g = xn.load(networkFile)
    entries = []
    if("CellType" not in g.vertex_attributes()):
        g.vs["CellType"] = g.vs["cell_type"]
    for resolution in (resolutions):
        g.vs["Leiden_weighted_%f"%resolution] = leiden(g,weights="weight",resolution=resolution)
        entries.append({
            "network": networkFile.stem,
            "parameter":resolution,
            "method":"Leiden",
            "level":0,
            "weighted":True,
        })
        entries[-1].update(calculateMetrics(g,"Leiden_weighted_%f"%resolution))

        g.vs["Leiden_unweighted_%f"%resolution] = leiden(g,weights=None,resolution=resolution)
        entries.append({
            "network": networkFile.stem,
            "parameter":resolution,
            "method":"Leiden",
            "level":0,
            "weighted":False,
        })
        entries[-1].update(calculateMetrics(g,"Leiden_unweighted_%f"%resolution))

    for markovTime in markovTimes:
        levelInfomapWeighted = infomapApply(g,markovTime,weights="weight")
        for levelIndex,levelData in enumerate(levelInfomapWeighted):
            g.vs["Infomap_weighted_%f_%d"%(markovTime,levelIndex)] = levelData
            entries.append({
                "network": networkFile.stem,
                "parameter":markovTime,
                "method":"Infomap",
                "level":levelIndex,
                "weighted":True,
            })
            entries[-1].update(calculateMetrics(g,"Infomap_weighted_%f_%d"%(markovTime,levelIndex)))
            
        levelInfomapUnweighted = infomapApply(g,markovTime,weights=None)
        for levelIndex,levelData in enumerate(levelInfomapUnweighted):
            g.vs["Infomap_unweighted_%f_%d"%(markovTime,levelIndex)] = levelData
            entries.append({
                "network": networkFile.stem,
                "parameter":markovTime,
                "method":"Infomap",
                "level":levelIndex,
                "weighted":False,
            })
            entries[-1].update(calculateMetrics(g,"Infomap_unweighted_%f_%d"%(markovTime,levelIndex)))

    levelNestedSBM = SBMMinimizeMembershipNested(g)
    for levelIndex,levelData in enumerate(levelNestedSBM):
        g.vs["SBMNested_unweighted_%d"%(levelIndex)] = levelData
        entries.append({
            "network": networkFile.stem,
            "parameter":0,
            "method":"SBMNested",
            "level":levelIndex,
            "weighted":False,
        })
        entries[-1].update(calculateMetrics(g,"SBMNested_unweighted_%d"%(levelIndex)))
    if(g.vcount()<20000):
        g.vs["SBM"] = SBMMinimizeMembershipWeighted(g)
        entries.append({
            "network": networkFile.stem,
            "parameter":0,
            "method":"SBM",
            "level":0,
            "weighted":True,
        })
        entries[-1].update(calculateMetrics(g,"SBM"))

    g.vs["SBM_unweighted"] = SBMMinimizeMembership(g)
    entries.append({
        "network": networkFile.stem,
        "parameter":0,
        "method":"SBM",
        "level":0,
        "weighted":False,
    })
    entries[-1].update(calculateMetrics(g,"SBM_unweighted"))

    # g.vs["OSLOM"],_ = oslom.oslomMembership(networkFile.stem,g.get_edgelist(),g.vcount(),g.is_directed(),weights=g.es["weight"])
    # entries.append({
    #     "network": networkFile.stem,
    #     "parameter":0,
    #     "method":"OSLOM",
    #     "level":0,
    #     "weighted":False,
    # })
    # entries[-1].update(calculateMetrics(g,"OSLOM"))


    xn.save(g,networksWithCommunitiesPath/networkFile.name)
    del g
    
    return entries

if __name__ == "__main__":
    allEntries = []
    networkFiles = list(networksPath.glob("*_pos.xnet"))
    # include seurat networks
    networkFiles += list(seuratNetworksPath.glob(f"{datasetName}_{variantName}_seurat_pos.xnet"))
    
    # shuffle
    np.random.shuffle(networkFiles)

    # for networkFile in tqdm(networkFiles[:1]):
    #     entries = processnetwork(networkFile)
    #     allEntries+=(entries)
    #     df = pd.DataFrame(allEntries)
    #     df.to_csv("network_communities.csv")
    # 
    # Change to multiprocessing
    with Pool(30) as p:
        for entries in tqdm(p.imap_unordered(processnetwork,networkFiles),total=len(networkFiles)):
            allEntries+=(entries)
            df = pd.DataFrame(allEntries)
            df.to_csv(f"network_communities_{datasetName}_{variantName}.csv")

# for networkFile in tqdm(networkFiles):
#     g = xn.load(networkFile)
#     if "CellType" not in g.vertex_attributes():
#         print(networkFile)
#         break
