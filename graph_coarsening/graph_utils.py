import numpy as np
import pygsp as gsp
import networkx as nx
from pygsp import graphs

def to_networkx():
    import networkx as nx
    return nx.from_scipy_sparse_matrix(G.W)
    
def get_neighbors(G, i):
    return G.A[i,:].indices
    # return np.arange(G.N)[np.array((G.W[i,:] > 0).todense())[0]]
    
def get_giant_component(G):

    from scipy.sparse import csgraph

    [ncomp, labels] = csgraph.connected_components(G.W, directed=False, return_labels=True)

    W_g = np.array((0,0))
    coords_g = np.array((0,2))
    keep = np.array(0)
    
    for i in range(0,ncomp):
        
        idx = np.where(labels!=i)
        idx = idx[0]
        
        if G.N-len(idx) > W_g.shape[0]:        
            W_g = G.W.toarray()
            W_g = np.delete(W_g, idx, axis=0)
            W_g = np.delete(W_g, idx, axis=1)
            if hasattr(G, 'coords'):
                coords_g = np.delete(G.coords, idx, axis=0)
            keep = np.delete(np.arange(G.N), idx)    

    if not hasattr(G, 'coords'):
        # print(W_g.shape)
        G_g = gsp.graphs.Graph(W=W_g)        
    else:
        G_g = gsp.graphs.Graph(W=W_g, coords=coords_g)

    
    return (G_g, keep)


def get_S(G):
    """
    Construct the N x |E| gradient matrix S
    """
    # the edge set
    edges = G.get_edge_list()
    weights = np.array(edges[2])
    edges = np.array(edges[0:2])
    M = edges.shape[1]
    
    # Construct the N x |E| gradient matrix S
    S = np.zeros((G.N,M))
    for e in np.arange(M):
        S[edges[0,e], e] = np.sqrt(weights[e])
        S[edges[1,e], e] = -np.sqrt(weights[e])
        
    return S   

# Compare the spectum of L and Lc
def eig(A, order='ascend'):

    # eigenvalue decomposition
    [l,X] = np.linalg.eigh(A)

    # reordering indices     
    idx = l.argsort()   
    if order == 'descend':
        idx = idx[::-1]

    # reordering     
    l = np.real(l[idx])
    X = X[:, idx]
    return (X,np.real(l))

def zero_diag(A):
    
    import scipy as sp

    if sp.sparse.issparse(A):
        # return A - sp.sparse.dia_matrix((A.diagonal()[sp.newaxis, :], [0]), shape=(A.shape[0], A.shape[1]))
        return A - sp.sparse.dia_matrix((A.diagonal()[np.newaxis, :], [0]), shape=(A.shape[0], A.shape[1]))
    else:
        D = A.diagonal()
        return A - np.diag(D)

def is_symmetric(As):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    As : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    from scipy import sparse 
    
    if As.shape[0] != As.shape[1]:
        return False

    if not isinstance(As, sparse.coo_matrix):
        As = sparse.coo_matrix(As)

    r, c, v = As.row, As.col, As.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check

def safe_setPropertiesToNodes(Gc, Call, G_original, tol=1e-12):
    """
    Set properties sizeSuperNode and nodesInSuperNode in the reduced graph,
    after filtering out nodes with degree 0 and validating consistency.

    Parameters
    ----------
    Gc : pygsp.graphs.Graph
        Reduced graph after coarsening.
    Call : list
        List of coarsening matrices per level.
    G_original : pygsp.graphs.Graph
        Original full graph.
    tol : float
        Tolerance to consider degree effectively zero.

    Returns
    -------
    nx_graph_H : networkx.Graph
        NetworkX graph with properties assigned correctly.
    new_graph : networkx.Graph
        Optional: Clean graph with remapped supernodes (most connected node as representative).
    """


    print("🔵 Starting safe_setPropertiesToNodes...")

    # 1. Eliminar nodos aislados
    degrees = np.array(Gc.W.sum(axis=1)).flatten()
    keep = degrees > tol
    if np.any(~keep):
        print(f"⚠️ Eliminando {np.sum(~keep)} nodos aislados del grafo reducido.")
        Wc = Gc.W[keep][:, keep]
        if hasattr(Gc, "coords"):
            coords = np.array(Gc.coords)[keep]
            Gc = graphs.Graph(Wc, coords=coords)
        else:
            Gc = graphs.Graph(Wc)
        Gc.original_node_ids = np.where(keep)[0]  # guardamos remapeo

    # 2. Construir NetworkX graph
    nx_graph_H = nx.from_scipy_sparse_array(Gc.W)
    if hasattr(Gc, "original_node_ids"):
        mapping = dict(enumerate(Gc.original_node_ids))
        nx_graph_H = nx.relabel_nodes(nx_graph_H, mapping)

    print(f"✅ Grafo reducido ahora tiene {nx_graph_H.number_of_nodes()} nodos y {nx_graph_H.number_of_edges()} aristas.")

    # 3. Asignar propiedades de supernodo
    print("🛠️ Asignando propiedades 'sizeSuperNode' y 'nodesInSuperNode'...")
    for node in nx_graph_H.nodes():
        nx_graph_H.nodes[node]['sizeSuperNode'] = 0
        nx_graph_H.nodes[node]['nodesInSuperNode'] = []

    # Reconstruir mapeo usando Call
    node_mapping = {i: [i] for i in range(len(Call[0].indices))}
    for level in range(len(Call)):
        new_node_mapping = {}
        for vi in range(len(Call[level].indices)):
            index = Call[level].indices[vi].real
            if index not in new_node_mapping:
                new_node_mapping[index] = []
            new_node_mapping[index].extend(node_mapping[vi])
        node_mapping = new_node_mapping

    # Asignar tamaño y miembros
    supernode_empty_count = 0
    for node, members in node_mapping.items():
        if node in nx_graph_H.nodes():
            nx_graph_H.nodes[node]['sizeSuperNode'] = len(members)
            nx_graph_H.nodes[node]['nodesInSuperNode'] = members
        else:
            supernode_empty_count += 1

    if supernode_empty_count > 0:
        print(f"⚠️ {supernode_empty_count} supernodos fueron descartados porque los nodos ya no existen (posiblemente grado 0).")

    # 4. Validación de consistencia
    print("🔍 Validando consistencia...")
    for node in nx_graph_H.nodes():
        members = nx_graph_H.nodes[node]['nodesInSuperNode']
        if isinstance(members, list) and len(members) == 0:
            print(f"⚠️ Supernodo {node} tiene 'nodesInSuperNode' vacío, posible inconsistencia.")

    print("✅ safe_setPropertiesToNodes terminado correctamente.")

    return nx_graph_H
