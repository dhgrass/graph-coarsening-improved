import numpy as np
import scipy as sp
import pygsp as gsp

import os
import tempfile
import zipfile

from . import graph_utils

from pygsp import graphs
from scipy import sparse
from urllib import request

import networkx as nx

_YEAST_URL = "http://nrvis.com/download/data/bio/bio-yeast.zip"
_MOZILLA_HEADERS = [("User-Agent", "Mozilla/5.0")]


def download_yeast():
    r"""
    A convenience method for loading a network of protein-to-protein interactions in budding yeast.

    http://networkrepository.com/bio-yeast.php
    """
    with tempfile.TemporaryDirectory() as tempdir:
        zip_filename = os.path.join(tempdir, "bio-yeast.zip")
        with open(zip_filename, "wb") as zip_handle:
            opener = request.build_opener()
            opener.addheaders = _MOZILLA_HEADERS
            request.install_opener(opener)
            with request.urlopen(_YEAST_URL) as url_handle:
                zip_handle.write(url_handle.read())
        with zipfile.ZipFile(zip_filename) as zip_handle:
            zip_handle.extractall(tempdir)
        mtx_filename = os.path.join(tempdir, "bio-yeast.mtx")
        with open(mtx_filename, "r") as mtx_handle:
            _ = next(mtx_handle)  # header
            n_rows, n_cols, _ = next(mtx_handle).split(" ")
            E = np.loadtxt(mtx_handle)
    E = E.astype(int) - 1
    W = sparse.lil_matrix((int(n_rows), int(n_cols)))
    W[(E[:, 0], E[:, 1])] = 1
    W = W.tocsr()
    W += W.T
    return W


def real(N, graph_name, connected=True):
    r"""
    A convenience method for loading toy graphs that have been collected from the internet.

	Parameters:
	----------
	N : int
	    The number of nodes. Set N=-1 to return the entire graph.

	graph_name : a string
        Use to select which graph is returned. Choices include
            * airfoil
                Graph from airflow simulation
                http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.9217&rep=rep1&type=pdf
                http://networkrepository.com/airfoil1.php
            * yeast
                Network of protein-to-protein interactions in budding yeast.
                http://networkrepository.com/bio-yeast.php
            * minnesota
                Minnesota road network.
                I am using the version provided by the PyGSP software package (initially taken from the MatlabBGL library.)
            * bunny
                The Stanford bunny is a computer graphics 3D test model developed by Greg Turk and Marc Levoy in 1994 at Stanford University
                I am using the version provided by the PyGSP software package.
	connected : Boolean
        Set to True if only the giant component is to be returned.
    """

    directory = os.path.join(
        os.path.dirname(os.path.dirname(graph_utils.__file__)), "data"
    )

    tries = 0
    while True:
        tries = tries + 1

        if graph_name == "airfoil":
            G = graphs.Airfoil()
            G = graphs.Graph(W=G.W[0:N, 0:N], coords=G.coords[0:N, :])

        elif graph_name == "yeast":
            W = download_yeast()
            G = graphs.Graph(W=W[0:N, 0:N])

        elif graph_name == "minnesota":
            G = graphs.Minnesota()
            W = G.W.astype(np.float)
            G = graphs.Graph(W=W[0:N, 0:N], coords=G.coords[0:N, :])

        elif graph_name == "bunny":
            G = graphs.Bunny()
            W = G.W.astype(np.float)
            G = graphs.Graph(W=W[0:N, 0:N], coords=G.coords[0:N, :])

        if connected == False or G.is_connected():
            break
        if tries > 1:
            print("WARNING: Disconnected graph. Using the giant component.")
            G, _ = graph_utils.get_giant_component(G)
            break
            
    if not hasattr(G, 'coords'): 
        try:
            import networkx as nx
            # graph = nx.from_scipy_sparse_matrix(G.W)
            graph = nx.from_scipy_sparse_array(G.W)
            pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')  
            G.set_coordinates(np.array(list(pos.values()))) 
        except ImportError:
            G.set_coordinates()
        
    return G

def to_pygsp_graph(nx_graph):
        """Converts a NetworkX graph to a PyGSP Graph."""
        W = nx.to_scipy_sparse_array(nx_graph, format="csr")
        G = graphs.Graph(W)
        if not hasattr(G, "coords") or G.coords is None:
            try:
                pos = nx.nx_agraph.graphviz_layout(nx_graph, prog="neato")
                G.set_coordinates(np.array(list(pos.values())))
            except ImportError:
                G.set_coordinates()
        return G

def bigNets(N, graph_name, connected=True):

    nx_graph = nx.read_gml("/home/darian/graph-coarsening/BigNets/{graph_name}.gml")
    G = to_pygsp_graph(nx_graph)

    # Recortar nodos si se especifica un límite N
    if N > 0 and N < G.N:
        G = graphs.Graph(W=G.W[0:N, 0:N])
        if hasattr(G, "coords") and G.coords is not None:
            G.set_coordinates(G.coords[0:N, :])

    # Asegurar conectividad si es necesario
    if connected and not G.is_connected():
        print(f"WARNING: {graph_name} is disconnected. Using the giant component.")
        G, _ = graph_utils.get_giant_component(G)

    return G


def realAcademic(N, graph_name, connected=True):
    """
    A convenience method for loading academic graphs.

    Parameters
    ----------
    N : int
        The maximum number of nodes to consider. Set N=-1 to return the entire graph.
    graph_name : str
        Use to select which academic graph is returned. Choices include:
            * karate_club
            * dolphins
            * political_books
            * football
    connected : bool
        Set to True if only the giant component is to be returned.

    Returns
    -------
    G : pygsp.graphs.Graph
        The loaded graph as a PyGSP Graph object.
    """

    
    
    print(f"Loading academic graph: {graph_name}")

    # Cargar grafos académicos
    if graph_name == "karate":
        nx_graph = nx.read_gml("/home/darian/graph-coarsening/academicNetworks_final_test/karate.gml")
        G = to_pygsp_graph(nx_graph)
        # nx_graph = nx.karate_club_graph()
        # for u, v, d in nx_graph.edges(data=True):
        #     if 'weight' in d:
        #         del d['weight']
        # G = to_pygsp_graph(nx_graph)

    elif graph_name == "dolphins":
        nx_graph = nx.read_gml("/home/darian/graph-coarsening/academicNetworks_final_test/dolphins.gml")
        G = to_pygsp_graph(nx_graph)

    elif graph_name == "polbooks":
        nx_graph = nx.read_gml("/home/darian/graph-coarsening/academicNetworks_final_test/polbooks.gml")
        G = to_pygsp_graph(nx_graph)

    elif graph_name == "football":
        nx_graph = nx.read_gml("/home/darian/graph-coarsening/academicNetworks_final_test/football.gml")
        G = to_pygsp_graph(nx_graph)

    else:
        raise ValueError(f"Graph name '{graph_name}' not recognized.")

    # Recortar nodos si se especifica un límite N
    if N > 0 and N < G.N:
        G = graphs.Graph(W=G.W[0:N, 0:N])
        if hasattr(G, "coords") and G.coords is not None:
            G.set_coordinates(G.coords[0:N, :])

    # Asegurar conectividad si es necesario
    if connected and not G.is_connected():
        print(f"WARNING: {graph_name} is disconnected. Using the giant component.")
        G, _ = graph_utils.get_giant_component(G)

    return G



def models(N, graph_name, connected=True, default_params=False, k=12, sigma=0.5):

    tries = 0
    while True:
        tries = tries + 1
        if graph_name == "regular":
            if default_params:
                k = 10
            offsets = []
            for i in range(1, int(k / 2) + 1):
                offsets.append(i)
                offsets.append(-(N - i))

            offsets = np.array(offsets)
            vals = np.ones_like(offsets)
            W = sp.sparse.diags(
                vals, offsets, shape=(N, N), format="csc", dtype=np.float
            )
            W = (W + W.T) / 2
            G = graphs.Graph(W=W)

        else:
            print("ERROR: uknown model")
            return

        if connected == False or G.is_connected():
            break
        if tries > 1:
            print("WARNING: disconnected graph.. trying to use the giant component")
            G = graph_utils.get_giant_component(G)
            break
    return G
