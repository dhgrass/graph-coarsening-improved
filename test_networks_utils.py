import glob
import os
import pprint
import pandas as pd
import matplotlib.pyplot as plt
from graph_coarsening import coarsening_utils
from graph_coarsening import graph_lib

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygsp import graphs  

from metrics import *
# Parámetros globales
r = 0.6  # coarsening ratio
methods = [
    "variation_neighborhoods",
    "variation_edges",
    "variation_cliques",
    "heavy_edge",
    "algebraic_JC",
    "affinity_GS",
    "kron",
]

def save_metrics_to_excel_allMethods(network_output_dir, graph_name):
    one_table_results = pd.DataFrame()

    for method in methods:
        # Cargar los resultados de las métricas desde el archivo Excel
        file_path = os.path.join(network_output_dir, f'{graph_name}_metrics_results_{method}.xlsx')
        df = pd.read_excel(file_path)
        
        # Concatenar los resultados en un solo DataFrame
        one_table_results = pd.concat([one_table_results, df], ignore_index=True)
    # Guardar el DataFrame combinado en un archivo Excel
    final_path = os.path.join(network_output_dir, f'{graph_name}_all_metrics_results.xlsx')
    one_table_results.to_excel(final_path, index=False)
    print(f'Results saved in name of network "graph_name" .... _all_metrics_results.xlsx')

def save_metrics_to_excel(metrics_list, filename):
    data = []
    for metrics in metrics_list:
        network_name = metrics['Network']
        method = metrics['Method']
        data.append([network_name, method, 'Spectral Ratio', metrics['Spectral Ratio (Original)'], metrics['Spectral Ratio (Reduced)']])
        data.append([network_name, method, 'Eigenratio', metrics['Eigenratio (Original)'], metrics['Eigenratio (Reduced)']])
        data.append([network_name, method, 'Spectral Gap', metrics['Spectral Gap (Original)'], metrics['Spectral Gap (Reduced)']])
        data.append([network_name, method, 'Algebraic Connectivity', metrics['Algebraic Connectivity (Original)'], metrics['Algebraic Connectivity (Reduced)']])
        data.append([network_name, method, 'Number of Nodes', metrics['Number of Nodes (Original)'], metrics['Number of Nodes (Reduced)']])
        data.append([network_name, method, 'Number of Edges', metrics['Number of Edges (Original)'], metrics['Number of Edges (Reduced)']])

    df = pd.DataFrame(data, columns=['Network', 'Method', 'Metric', 'Original', 'Reduced'])
    df.to_excel(filename, index=False)

def save_adjacency_matrix_to_file(nets_path, output_dir):
    """
    Guarda la matriz de adyacencia en un archivo de texto.
    """

    # Obtener la lista de archivos .gml en el directorio pathBignets
    graph_files = glob.glob(os.path.join(nets_path, '*.gml'))

    for graph_file in graph_files:
        graph_name = os.path.splitext(os.path.basename(graph_file))[0]
        print(f"\nEvaluando grafo: {graph_name}")
        network_output_dir = os.path.join(output_dir, graph_name)

        # Cargar el grafo desde el archivo .gml
        nx_graph = nx.read_gml(graph_file)

        adj_matrix_G = nx.to_numpy_array(nx_graph, weight=None)
        adj_matrix_G_filename = os.path.join(network_output_dir, f'{graph_name}_original_adj_matrix.txt')
        with open(adj_matrix_G_filename, 'w') as f:
            for row in adj_matrix_G:
                f.write(' '.join(map(str, row.astype(int))) + '\n')
        print(f"Matriz de adyacencia guardada en: {adj_matrix_G_filename}")

def filter_nodes_with_degree_zero(G, tol=1e-12):
    degrees = np.array(G.W.sum(axis=1)).flatten()
    keep = degrees > tol
    if np.any(~keep):
        print(f"⚠️ Eliminando {np.sum(~keep)} nodos aislados tras filtrado (tol={tol})")
        W_filtered = G.W[keep][:, keep]
        if hasattr(G, "coords"):
            coords_filtered = np.array(G.coords)[keep]
            G_filtered = graphs.Graph(W_filtered, coords=coords_filtered)
        else:
            G_filtered = graphs.Graph(W_filtered)
        G_filtered.original_node_ids = np.where(keep)[0]
        return G_filtered
    else:
        return G  # Si no había nodos aislados, devolvés el mismo grafo
