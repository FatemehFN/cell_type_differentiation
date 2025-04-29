import time
import numpy as np
import pandas as pd
import clusim.sim as csim
import leidenalg
from igraph import Graph
from clusim.clustering import Clustering
import xnetwork as xn

def cell_type_dict(ROSMAP_graph):
    file_path = '/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/68k_pbmc_barcodes_annotation.tsv'
    df = pd.read_csv(file_path, sep='\t')
    barcode_celltype_dict = dict(zip(df['barcodes'], df['celltype']))
    unique_cell_types = df['celltype'].unique().tolist()
    print('The number of cell types is', len(unique_cell_types))

    membership_vector = []
    cell_type_strings = []
    for node_name in ROSMAP_graph.vs['name']:
        membership_vector.append(unique_cell_types.index(barcode_celltype_dict[node_name]))
        cell_type_strings.append(barcode_celltype_dict[node_name])
    return membership_vector, cell_type_strings

def leiden_best_q(ROSMAP_graph, cell_type_membership_vector, res):
    best_q = 0
    best_match = 0
    # Run the partitioning 100 times and keep the best based on quality
    for i in range(1):
        partition_ROSMAP = leidenalg.find_partition(ROSMAP_graph, leidenalg.RBConfigurationVertexPartition,
                                                      resolution_parameter=res)
        fromCluster = Clustering()
        fromCluster.from_membership_list(partition_ROSMAP.membership)
        toCluster = Clustering()
        toCluster.from_membership_list(cell_type_membership_vector)
        ariValue = csim.adjrand_index(fromCluster, toCluster)
        q = partition_ROSMAP.quality()
        if q > best_q:
            # Save the best partition as a string attribute on vertices.
            ROSMAP_graph.vs["comm_leiden"] = [str(entry) for entry in partition_ROSMAP.membership]
            best_match = ariValue
            best_q = q
    return best_match, ROSMAP_graph


def leiden_best_q_weighted(ROSMAP_graph,weights,cell_type_membership_vector):

    best_q=0
    best_match=0
    res=0.01
    for i in range(1):
        partition_ROSMAP = leidenalg.find_partition(ROSMAP_graph,leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=res,weights=weights)


        
        fromCluster = Clustering()
        fromCluster.from_membership_list(partition_ROSMAP.membership)
        toCluster = Clustering()
        toCluster.from_membership_list(cell_type_membership_vector)


        ariValue = csim.adjrand_index(fromCluster,toCluster)
        q=partition_ROSMAP.quality()
        if q>best_q:
            ROSMAP_graph.vs["comm_leiden"] = [str(entry) for entry in partition_ROSMAP.membership]
            best_match=ariValue
            best_q=q
    
    return best_match,ROSMAP_graph

def main():
    # --- Load Graph ---
    ROSMAP_graph = Graph(directed=False)
    edge_file = '/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/graph_edge_list_68kpbmc_weighted.txt'
    with open(edge_file, 'r') as f:
        lines = f.readlines()

    edges = []
    repetitive_nodes = []
    weights = []
    for item in lines[1:]:
        parts = item.split('\t')
        from_node = parts[0].strip('""')
        to_node = parts[1].strip('""')
        weight_val = float(parts[-1].rstrip())
        edges.append((from_node, to_node))
        repetitive_nodes.extend([from_node, to_node])
        weights.append(weight_val)

    vertices_to_add = sorted(list(set(repetitive_nodes)))
    ROSMAP_graph.add_vertices(vertices_to_add)

    # --- Create the cell type membership vector ---
    ground_truth_cell_type_membership_vector, cell_type_strings = cell_type_dict(ROSMAP_graph)
    ROSMAP_graph.vs["cell_type"] = cell_type_strings

    # --- Add edges and weights ---
    ROSMAP_graph.add_edges(edges)
    ROSMAP_graph.es["weight"] = weights

    # --- Get connected components ---
    components = ROSMAP_graph.connected_components()
    num_components = len(components)
    print('Connected components:', num_components)

    # --- Set the resolution parameter ---
    res = 0.01  # You can change this to any resolution value as needed

    # --- Time measurement ---
    times = []
    for i in range(100):
        start_time = time.time()
        #score, ROSMAP_graph = leiden_best_q(ROSMAP_graph, ground_truth_cell_type_membership_vector, res)
        score,ROSMAP_graph=leiden_best_q_weighted(ROSMAP_graph,weights,ground_truth_cell_type_membership_vector)
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        print(f"Run {i+1}/100: {run_time:.4f} seconds")

    mean_time = np.mean(times)
    std_time = np.std(times)
    print("Mean time over 100 runs:", mean_time)
    print("Standard deviation:", std_time)

    # --- Save time data into a CSV file ---
    time_data = pd.DataFrame({'run': list(range(1, 101)), 'time_sec': times})
    time_data.to_csv('/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/timing_data.csv', index=False)

if __name__ == "__main__":
    main()
