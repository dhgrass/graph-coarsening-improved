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

from graph_coarsening.graph_utils import safe_setPropertiesToNodes
from metrics import *
from test_networks_utils import filter_nodes_with_degree_zero, save_metrics_to_excel, save_metrics_to_excel_allMethods
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

# Evaluar cada grafo académico
graph_names = ["karate", "dolphins", "polbooks", "football"]
# graph_names = ["karate"]

# Definir el directorio de salida
output_dir = '/home/darian/graph-coarsening/academicNetworks_final_test_RESULT/'

def test_academics_nets():
    """
    Test the academics networks
    """

    spectral_metrics_all = []
    
    tol = 1e-12

    for graph_name in graph_names:
        print(f"\nEvaluando grafo: {graph_name}")

        # Cargar el grafo utilizando realAcademic
        G = graph_lib.realAcademic(N=-1, graph_name=graph_name, connected=True)

        nx_graph = nx.from_scipy_sparse_array(G.W)

        # Calcular base espectral para operaciones avanzadas
        G.compute_fourier_basis()

        # Crear un directorio para la red si no existe
        network_output_dir = os.path.join(output_dir, graph_name)
        os.makedirs(network_output_dir, exist_ok=True)

        # Info general
        is_connected = nx.is_connected(nx_graph) if not nx.is_directed(nx_graph) else nx.is_weakly_connected(nx_graph)
        num_components = nx.number_connected_components(nx_graph) if not nx.is_directed(nx_graph) else nx.number_weakly_connected_components(nx_graph)
        is_directed = nx.is_directed(nx_graph)
        is_weighted = nx.is_weighted(nx_graph)

        print(f"    ¿Es conectado?: {'Sí' if is_connected else 'No'}")
        print(f"    Número de componentes conexas: {num_components}")
        print(f"    ¿Es dirigido?: {'Sí' if is_directed else 'No'}")
        print(f"    ¿Es ponderado?: {'Sí' if is_weighted else 'No'}")

        for method in methods:
            print(f"    - Método: {method}")

            # 4. Aplicar coarsening
            _, Gc, Call, _ = coarsening_utils.coarsen(G, r=0.6, method=method)

            # 6. Filtrar nodos de grado 0 con tolerancia
            Gc = filter_nodes_with_degree_zero(Gc, tol=tol)

            # 7. Crear nx_graph_H y propiedades de supernodos de forma segura
            nx_graph_H = safe_setPropertiesToNodes(Gc, Call, G, tol=tol)

            # 7. Convertir grafos a matrices de adyacencia
            adj_matrix_G = nx.to_numpy_array(nx_graph, weight=None)
            adj_matrix_H = nx.to_numpy_array(nx_graph_H, weight=None)

            # 8. Guardar matriz reducida
            adj_matrix_H_filename = os.path.join(network_output_dir, f'{graph_name}_{method}_reduced_adj_matrix.txt')
            with open(adj_matrix_H_filename, 'w') as f:
                for row in adj_matrix_H:
                    f.write(' '.join(map(str, row.astype(int))) + '\n')
            print(f"    Matriz de adyacencia guardada en: {adj_matrix_H_filename}")

            # 9. Calcular métricas espectrales
            spectral_metrics = analyze_spectral_properties(adj_matrix_G, adj_matrix_H)
            spectral_metrics['Network'] = graph_name
            spectral_metrics['Method'] = method
            spectral_metrics_all.append(spectral_metrics)

            # 10. Guardar resultados parciales por método
            metrics_result = os.path.join(network_output_dir, f'{graph_name}_metrics_results_{method}.xlsx')
            save_metrics_to_excel(spectral_metrics_all, metrics_result)
            spectral_metrics_all = []

        # 11. Guardar resultados combinados para el grafo
        # save_metrics_to_excel_allMethods(network_output_dir, graph_name)

    print("\n✅ Test de todas las redes terminado correctamente.")





if __name__ == "__main__":
    print("Starting...")

    test_academics_nets()    
    
    print("Done!")