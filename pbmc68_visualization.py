
from igraph import Graph
import pandas as pd
import xnetwork as xn

ROSMAP_graph=Graph(directed=False)
# f=open('/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/graph_edge_list_sample_2000_pbmc68k_weighted.txt','r')
f=open('/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/graph_edge_list_20k_68kpbmc_weighted.txt','r')

lines=f.readlines()
f.close()



def cell_type_dict(ROSMAP_graph):
     
    file_path='/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/68k_pbmc_barcodes_annotation.tsv'
    df = pd.read_csv(file_path, sep='\t')
    barcode_celltype_dict = dict(zip(df['barcodes'], df['celltype']))

    unique_cell_types = df['celltype'].unique().tolist()

    print('the number of cell types is',len(unique_cell_types))

    membership_vector=[]
    for node_name in ROSMAP_graph.vs['name']:
        membership_vector.append(unique_cell_types.index(barcode_celltype_dict[node_name]))

    return membership_vector


def positions_dict(ROSMAP_graph):
     
    file_path='/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/68k_pbmc_barcodes_annotation.tsv'
    df = pd.read_csv(file_path, sep='\t')
    positions = zip(df['TSNE.1'], df['TSNE.2'])

    position_dict = dict(zip(df['barcodes'], positions))
    
    membership_vector=[]
    for node_name in ROSMAP_graph.vs['name']:
        membership_vector.append(position_dict[node_name])

    return membership_vector



edges=[]
repetitive_nodes=[]
weights=[]
for item in lines[1:]:

    from_node=item.split('\t')[0].strip('""')
    to_node=item.split('\t')[1].strip('""')
    edges.append((from_node,to_node))
    repetitive_nodes.append(from_node)
    repetitive_nodes.append(to_node)
    weights.append(float(item.split('\t')[-1].rstrip()))


vertices_to_add=sorted(list(set(repetitive_nodes)))
ROSMAP_graph.add_vertices(vertices_to_add)


#--------make the membership vector---------
ground_truth_cell_type_membership_vector=cell_type_dict(ROSMAP_graph)


ROSMAP_graph.vs["cell_type"] = ground_truth_cell_type_membership_vector

ROSMAP_graph.vs["Position"] = positions_dict(ROSMAP_graph)

ROSMAP_graph.add_edges(edges)

ROSMAP_graph.es["weight"] = weights

# partition_ROSMAP = leidenalg.find_partition(ROSMAP_graph,leidenalg.RBConfigurationVertexPartition,
#                             resolution_parameter=0.1,weights='weight')



# ROSMAP_graph.vs["Leiden"] = partition_ROSMAP.membership


# xn.save(ROSMAP_graph,'/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/graph_edge_list_sample_2000_pbmc68k_weighted.xnet')
xn.save(ROSMAP_graph,'/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/graph_edge_list_20k_68kpbmc_weighted.xnet')