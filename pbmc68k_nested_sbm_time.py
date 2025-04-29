from pathlib import Path
import graph_tool as gt
import graph_tool.inference as gtInference
from graph_tool import Graph as gtGraph
import matplotlib
from clusim.clustering import Clustering, print_clustering
from collections import Counter
import pandas as pd
import numpy as np
import clusim.sim as csim
from mpmath import mp  # pip install mpmath
from igraph import Graph

# Set up mpmath precision
mp.dps = 50

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def cell_type_dict(ROSMAP_graph):
    file_path = '/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/68k_pbmc_barcodes_annotation.tsv'
    df = pd.read_csv(file_path, sep='\t')
    barcode_celltype_dict = dict(zip(df['barcodes'], df['celltype']))
    unique_cell_types = df['celltype'].unique().tolist()
    
    membership_vector = []
    for node_name in ROSMAP_graph.vs['name']:
        membership_vector.append(unique_cell_types.index(barcode_celltype_dict[node_name]))
    return membership_vector

def SBMMinimizeMembershipNested(graph, weights=None, weightMode="real-normal"):
    vertexCount = graph.vcount()
    # Create a graph_tool Graph
    g = gt.Graph(directed=graph.is_directed())
    for _ in range(vertexCount):
        g.add_vertex()

    weighted = bool(weights)
    if weights:
        weightsProperty = g.new_edge_property("double")

    for edge in graph.es:
        gedge = g.add_edge(edge.source, edge.target)
        if weighted:
            weight = edge[weights]
            weightsProperty[gedge] = weight
    if weighted:
        g.edge_properties[weights] = weightsProperty

    state_args = {}
    if weighted:
        state_args["recs"] = [g.ep.weight]
        state_args["rec_types"] = [weightMode]

    state_args["deg_corr"] = True
    state = gtInference.minimize.minimize_nested_blockmodel_dl(
        g, state_args=state_args
    )
    
    dl = state.entropy()
    levelMembership = []
    levels = state.get_levels()
    lastIndices = list(range(vertexCount))
    for level, s in enumerate(levels):
        blocks = s.get_blocks()
        lastIndices = [blocks[gindex] for gindex in lastIndices]
        if len(set(lastIndices)) == 1 and level != 0:
            break
        levelMembership.append([str(entry) for entry in lastIndices])
  
    levelMembership.reverse()
    return levelMembership, dl

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
        edges.append((from_node, to_node))
        repetitive_nodes.extend([from_node, to_node])
        weights.append(float(parts[-1].rstrip()))
    
    vertices_to_add = sorted(list(set(repetitive_nodes)))
    ROSMAP_graph.add_vertices(vertices_to_add)
    
    # --- Add edges and weights ---
    ROSMAP_graph.add_edges(edges)
    ROSMAP_graph.es["weight"] = weights

    # --- Create the ground truth cell type membership vector ---
    ground_truth_cell_type_membership_vector = cell_type_dict(ROSMAP_graph)

    # --- Run nested SBM 20 times and record timing info ---
    iterations = 10
    timing_results = []
    
    print("Running nested SBM (weighted) 10 times...")
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        start_time = pd.Timestamp.now()
        memberships, dl = SBMMinimizeMembershipNested(ROSMAP_graph, weights='weight')
        #memberships, dl = SBMMinimizeMembershipNested(ROSMAP_graph)
        end_time = pd.Timestamp.now()
        run_time = (end_time - start_time).total_seconds()
        timing_results.append({'iteration': i+1, 'time_sec': run_time})
        print(f"Iteration {i+1} completed in {run_time:.4f} seconds")
    
    times = [entry['time_sec'] for entry in timing_results]
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Mean time over {iterations} runs: {mean_time:.4f} seconds")
    print(f"Standard deviation: {std_time:.4f} seconds")
    
    # --- Save timing data into a CSV file ---
    timing_df = pd.DataFrame(timing_results)
    output_csv = '/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/nested_sbm_timing_data.csv'
    timing_df.to_csv(output_csv, index=False)
    print(f"Timing data saved to {output_csv}")

if __name__ == "__main__":
    main()
