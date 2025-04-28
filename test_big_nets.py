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
from test_networks_utils import save_metrics_to_excel_allMethods, setPropertiesToNodes

# Parámetros globales
r = 0.6  # coarsening ratio
methods = [
    # "variation_neighborhoods",
    # "variation_edges",
    # "variation_cliques",
    # "heavy_edge",
    # "algebraic_JC",
    # "affinity_GS",
    "kron",
]

big_nets_path = os.path.join(os.getcwd(), "temporal_BigNets")
output_dir = os.path.join(os.getcwd(), "temporal_BigNets_RESULT")
# big_nets_path = os.path.join(os.getcwd(), "bigsNetworks_final_test")
# output_dir = os.path.join(os.getcwd(), "bigsNetworks_final_test_RESULT")




def save_adjacency_matrix_to_file():
    """
    Guarda la matriz de adyacencia en un archivo de texto.
    """

    # Obtener la lista de archivos .gml en el directorio pathBignets
    graph_files = glob.glob(os.path.join(big_nets_path, '*.gml'))

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

def testBigNets():
    
    spectral_metrics_all = []
    # Obtener la lista de archivos .gml en el directorio pathBignets
    graph_files = glob.glob(os.path.join(big_nets_path, '*.gml'))

    tol = 1e-12

    for graph_file in graph_files:
        graph_name = os.path.splitext(os.path.basename(graph_file))[0]
        print(f"\nEvaluando grafo: {graph_name}")

        # Cargar el grafo desde el archivo .gml
        nx_graph = nx.read_gml(graph_file)

        # Verificar si el grafo es conectado o no
        is_connected_nx_graph = nx.is_connected(nx_graph) if not nx.is_directed(nx_graph) else nx.is_weakly_connected(nx_graph)
        print(f"    ¿Es conectado?: {'Sí' if is_connected_nx_graph else 'No'}")

        # Contar el número de componentes conexas
        num_components_nx_graph = nx.number_connected_components(nx_graph) if not nx.is_directed(nx_graph) else nx.number_weakly_connected_components(nx_graph)
        print(f"    Número de componentes conexas: {num_components_nx_graph}")

        # Verificar si el grafo es dirigido o no
        is_directed_nx_graph = nx.is_directed(nx_graph)
        print(f"    ¿Es dirigido?: {'Sí' if is_directed_nx_graph else 'No'}")

        # Verificar si el grafo es ponderado o no
        is_weighted_nx_graph = nx.is_weighted(nx_graph)
        print(f"    ¿Es ponderado?: {'Sí' if is_weighted_nx_graph else 'No'}")

        # Convertir el grafo de NetworkX a la estructura esperada por graph_lib
        G = graph_lib.to_pygsp_graph(nx_graph)

        # Calcular base espectral para operaciones avanzadas
        G.compute_fourier_basis()

        # Crear un directorio para la red si no existe
        network_output_dir = os.path.join(output_dir, graph_name)
        os.makedirs(network_output_dir, exist_ok=True)

        for method in methods:
            print(f"    - Método: {method}")

            # Aplicar coarsening
            _, Gc, Call, _ = coarsening_utils.coarsen(G, r=r, method=method) # en Call está lo que necesito, propiedad indices

            # ⚠️ Generar nx_graph_H directamente desde Gc antes de filtrar
            nx_graph_H = nx.from_scipy_sparse_array(Gc.W)

            # Número de componentes conexas
            components_original = nx.number_connected_components(nx_graph)
            components_reduced = nx.number_connected_components(nx_graph_H)

            # Número de nodos con grado 0
            degree_dict_G = dict(nx.degree(nx_graph))
            nodes_deg0_G = sum(1 for _, d in degree_dict_G.items() if d == 0)

            degree_dict_H = dict(nx.degree(nx_graph_H))
            nodes_deg0_H = sum(1 for _, d in degree_dict_H.items() if d == 0)

            print(f"    Componentes conexas: Original = {components_original}, Reducido = {components_reduced}")
            print(f"    Nodos grado 0:       Original = {nodes_deg0_G}, Reducido = {nodes_deg0_H}")


            # acá tengo que crear las propiedades 'sizeSuperNode' y 'nodesInSuperNode' en nx_graph_H 
            _, G_reducido_update = setPropertiesToNodes(nx_graph_H, Call, G)


            # 🔧 FILTRAR NODOS CON GRADO 0
            Gc = filter_nodes_with_degree_zero(Gc)

            if hasattr(Gc, "original_node_ids"):
                mapping = dict(enumerate(Gc.original_node_ids))
                nx_graph_H = nx.from_scipy_sparse_array(Gc.W)
                nx_graph_H = nx.relabel_nodes(nx_graph_H, mapping)
            else:
                nx_graph_H = nx.from_scipy_sparse_array(Gc.W)

            # nx_graph_H = nx.from_scipy_sparse_array(Gc.W)
            # _, G_reducido_update = setPropertiesToNodes(nx_graph_H, Call, G)

            # Convertir los grafos a matrices de adyacencia
            adj_matrix_G = nx.to_numpy_array(nx_graph, weight=None)
            adj_matrix_H = nx.to_numpy_array(nx_graph_H, weight=None)

            # Guardar la matriz de adyacencia adj_matrix_H en un archivo .txt iterando fila por fila
            adj_matrix_H_filename = os.path.join(network_output_dir, f'{graph_name}_{method}_reduced_adj_matrix.txt')
            with open(adj_matrix_H_filename, 'w') as f:
                for row in adj_matrix_H:
                    f.write(' '.join(map(str, row.astype(int))) + '\n')
            print(f"Matriz de adyacencia guardada en: {adj_matrix_H_filename}")

            # Calcular las métricas espectrales para el grafo original y el grafo reducido
            spectral_metrics = analyze_spectral_properties(adj_matrix_G, adj_matrix_H)

            spectral_metrics['Network'] = graph_name
            spectral_metrics['Method'] = method

            spectral_metrics_all.append(spectral_metrics)

            metrics_result = os.path.join(network_output_dir, f'{graph_name}_metrics_results_{method}.xlsx')
            save_metrics_to_excel(spectral_metrics_all, metrics_result)
            spectral_metrics_all = []

        # Guardar todos los resultados en un solo archivo Excel
        save_metrics_to_excel_allMethods(network_output_dir, graph_name)



def filter_nodes_with_degree_zero(G):
    degrees = np.array(G.W.sum(axis=1)).flatten()
    keep = degrees > 0
    original_node_ids = np.where(keep)[0]  # ← este es el mapeo

    W_filtered = G.W[keep][:, keep]

    if hasattr(G, "coords"):
        coords = np.array(G.coords)[keep]
        G_filtered = graphs.Graph(W_filtered, coords=coords)
    else:
        G_filtered = graphs.Graph(W_filtered)

    G_filtered.original_node_ids = original_node_ids  # ← guardamos el mapeo como atributo
    return G_filtered



if __name__ == "__main__":
    print("Starting Big nets...")

    testBigNets()    
    # save_adjacency_matrix_to_file()
    
    print("Big nets .... Done!")