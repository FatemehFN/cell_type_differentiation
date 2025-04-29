import time
import numpy as np
import pandas as pd
import clusim.sim as csim
from igraph import Graph
from clusim.clustering import Clustering, print_clustering
from infomap import Infomap
import xnetwork as xn

def cell_type_dict(ROSMAP_graph):
    file_path = '/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/68k_pbmc_barcodes_annotation.tsv'
    df = pd.read_csv(file_path, sep='\t')
    barcode_celltype_dict = dict(zip(df['barcodes'], df['celltype']))
    unique_cell_types = df['celltype'].unique().tolist()
    membership_vector = []
    for node_name in ROSMAP_graph.vs['name']:
        membership_vector.append(unique_cell_types.index(barcode_celltype_dict[node_name]))
    return membership_vector

def infomapApply(g, m_time, weights=None):
    vertexCount = g.vcount()
    if weights:
        edges = [(e.source, e.target, e[weights]) for e in g.es]
    else:
        edges = g.get_edgelist()

    extraOptions = "-d" if g.is_directed() else ""
    im = Infomap(f"{extraOptions} -N 1 --silent --seed {np.random.randint(4294967296)}", markov_time=m_time)
    im.setVerbosity(0)
    
    for nodeIndex in range(vertexCount):
        im.add_node(nodeIndex)
    for edge in edges:
        if len(edge) > 2:
            if edge[2] > 0:
                im.add_link(edge[0], edge[1], weight=edge[2])
        else:
            im.add_link(edge[0], edge[1])
    
    im.run()
    # Get multilevel modules and create a membership string per node.
    memberships = [":".join([str(a) for a in membership])
                   for index, membership in im.get_multilevel_modules().items()]
    levelCount = max([len(element.split(":")) for element in memberships])
    levelMembership = []
    for level in range(levelCount):
        levelMembership.append(
            [":".join(element.split(":")[:(level+1)]) for element in memberships]
        )
    return levelMembership, im.codelength

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
    
    # --- Infomap timing ---
    markov_time = 5  # Fixed markov time
    iterations = 100
    timing_results = []
    times = []
    
    print(f"Running Infomap with markov time {markov_time} for {iterations} iterations...")
    for i in range(iterations):
        start_time = time.time()
        memberships, codelength = infomapApply(ROSMAP_graph,weights='weight', m_time=markov_time)
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        timing_results.append({'markov_time': markov_time, 'run': i+1, 'time_sec': run_time})
        print(f"Run {i+1}/{iterations}: {run_time:.4f} seconds")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Mean time: {mean_time:.4f} seconds")
    print(f"Standard deviation: {std_time:.4f} seconds")
    
    # --- Save timing data into a CSV file ---
    timing_df = pd.DataFrame(timing_results)
    output_csv = '/mnt/adsinglecell/fatemeh/rosmapnetworks/pbmc68k/infomap_timing_data.csv'
    timing_df.to_csv(output_csv, index=False)
    print(f"Timing data saved to {output_csv}")

if __name__ == "__main__":
    main()
