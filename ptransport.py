import heapq
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt


def _compute_tangent_spaces(X, nnbrs, K, d):
    kn_graph = nnbrs.kneighbors_graph(n_neighbors=K).toarray()
    kn_graph -= np.eye((kn_graph.shape[0]))

    geodesic_neighborhoods = np.array([(X[np.nonzero(kn_graph[i])] - X[i]).T for i in tqdm(range(X.shape[0]))])
    u, _, _ = np.linalg.svd(geodesic_neighborhoods)

    return u[:, :, :d]



def _get_neigh_idxs_and_dists(nnbrs, make_graph_how):
    neigh_idxs = []
    if make_graph_how == 'neighbors':
        nnbrs_graph = nnbrs.kneighbors_graph().toarray()
        nnbrs_graph -= np.eye(nnbrs_graph.shape[0])
        nnbrs_idxs = [np.nonzero(nnbrs_graph[i])[0] for i in range(nnbrs_graph.shape[0])]
        
        nnbrs_dists = nnbrs.kneighbors_graph(mode='distance').toarray()
        
        return nnbrs_idxs, nnbrs_dists
    if make_graph_how == 'radius':
        raise NotImplementedError



def PTU_dists(
    X, 
    n_components, 
    radius = 1.0, 
    n_neighbors = 5, 
    make_graph_how = 'neighbors',
    K=None,
    n_jobs=-1):
    
    if make_graph_how != 'neighbors' and make_graph_how != 'radius':
        raise ValueError(f"Undefined mode {make_graph_how}. Possible variants: \'neighbors\' or \'radius\'")

    if K is None:
        K = n_neighbors

    n = X.shape[0]

    nnbrs = NearestNeighbors(
        n_neighbors=n_neighbors, 
        radius=radius,
        n_jobs=n_jobs)
    nnbrs.fit(X)
    print("NearestNeighbors fitted")
    tangent_spaces = _compute_tangent_spaces(X, nnbrs, K, n_components)
    print("tangent spaces computed")
    P = []
    j_used = [False] * n

    R = np.zeros((n, n_components, n_components))
    v = np.zeros((n, n_components))
    D = np.zeros((n, n))

    nnbrs_idxs, nnbrs_pairwise_dists = _get_neigh_idxs_and_dists(nnbrs, make_graph_how)

    for i in tqdm(range(n)):
        pred = np.zeros((n), dtype=int)
        dist = np.ones((n)) * np.infty # ??
        dist[i] = 0

        R[i] = np.eye(n_components)
        v[i] = np.zeros((n_components))
        

        for j in nnbrs_idxs[i]:
            pred[j] = int(i)
            dist[j] = nnbrs_pairwise_dists[j, i]
            heapq.heappush(P, (dist[j], j))
            j_used[j] = True
        
        while P: # while P is not empty
            item_r = heapq.heappop(P)
            j_used[item_r[1]] = False
            r = item_r[1]
            q = pred[r]
            u, s, vh = np.linalg.svd(tangent_spaces[q].T @ tangent_spaces[r])
            
            R[r] = R[q] @ u @ vh
            v[r] = v[q] + R[q] @ tangent_spaces[q].T @ (X[r] - X[q])
            D[i, r] = np.linalg.norm(v[r]) # geo_dist[r]
            
            for j in nnbrs_idxs[r]:
                tmp_dist = dist[r] + nnbrs_pairwise_dists[j, r]
                if tmp_dist < dist[j]:
                    dist[j] = tmp_dist
                    pred[j] = r
                    if not j_used[j]:
                        heapq.heappush(P, (dist[j], j))
                        j_used[j] = True
    D = (D + D.T) / 2
    return D

def PTU(
    X, 
    n_components, 
    radius = 1.0, 
    n_neighbors = 5, 
    make_graph_how = 'neighbors',
    K=None,
    n_jobs=-1):
    
    d = PTU_dists(X, n_components, radius, n_neighbors, make_graph_how, K, n_jobs)
    
    mds = MDS(n_components, dissimilarity='precomputed')
    embedded = mds.fit_transform(d)
    return embedded



def iso_dists(
    X, 
    n_components, 
    radius = 1.0, 
    n_neighbors = 5, 
    make_graph_how = 'neighbors',
    K=None,
    n_jobs=-1):
    
    if make_graph_how != 'neighbors' and make_graph_how != 'radius':
        raise ValueError(f"Undefined mode {make_graph_how}. Possible variants: \'neighbors\' or \'radius\'")

    if K is None:
        K = n_neighbors

    n = X.shape[0]

    nnbrs = NearestNeighbors(
        n_neighbors=n_neighbors, 
        radius=radius,
        n_jobs=n_jobs)
    nnbrs.fit(X)
    print("NearestNeighbors fitted")
    tangent_spaces = _compute_tangent_spaces(X, nnbrs, K, n_components)
    print("tangent spaces computed")
    P = []
    j_used = [False] * n

    # R = np.zeros((n, n_components, n_components))
    # v = np.zeros((n, n_components))
    D = np.zeros((n, n))

    nnbrs_idxs, nnbrs_pairwise_dists = _get_neigh_idxs_and_dists(nnbrs, make_graph_how)

    for i in tqdm(range(n)):
        pred = np.zeros((n), dtype=int)
        dist = np.ones((n)) * np.infty # ??
        dist[i] = 0

        # R[i] = np.eye(n_components)
        # v[i] = np.zeros((n_components))
        

        for j in nnbrs_idxs[i]:
            pred[j] = int(i)
            dist[j] = nnbrs_pairwise_dists[j, i]
            heapq.heappush(P, (dist[j], j))
            j_used[j] = True
        
        while P: # while P is not empty
            item_r = heapq.heappop(P)
            j_used[item_r[1]] = False
            r = item_r[1]
            q = pred[r]
            # u, s, vh = np.linalg.svd(tangent_spaces[q].T @ tangent_spaces[r])
            
            # R[r] = R[q] @ u @ vh
            # v[r] = v[q] + R[q] @ tangent_spaces[q].T @ (X[r] - X[q])
            D[i, r] = D[i, q] + nnbrs_pairwise_dists[j, r]#np.linalg.norm(v[r]) # geo_dist[r]
            
            for j in nnbrs_idxs[r]:
                tmp_dist = dist[r] + nnbrs_pairwise_dists[j, r]
                if tmp_dist < dist[j]:
                    dist[j] = tmp_dist
                    pred[j] = r
                    if not j_used[j]:
                        heapq.heappush(P, (dist[j], j))
                        j_used[j] = True
    D = (D + D.T) / 2
    return D

def dijkstra(neighbors, ledges, n):
    P = []
    j_used = [False] * n

    # R = np.zeros((n, n_components, n_components))
    # v = np.zeros((n, n_components))
    D = np.zeros((n, n))

    # nnbrs_idxs, nnbrs_pairwise_dists = _get_neigh_idxs_and_dists(nnbrs, make_graph_how)
    nnbrs_idxs, nnbrs_pairwise_dists = neighbors, ledges

    for i in tqdm(range(n)):
        pred = np.zeros((n), dtype=int)
        dist = np.ones((n)) * np.infty # ??
        dist[i] = 0
        
        # R[i] = np.eye(n_components)
        # v[i] = np.zeros((n_components))
        

        for j in nnbrs_idxs[i]:
            pred[j] = int(i)
            dist[j] = nnbrs_pairwise_dists[j, i]
            heapq.heappush(P, (dist[j], j))
            j_used[j] = True
        
        while P: # while P is not empty
            item_r = heapq.heappop(P)
            j_used[item_r[1]] = False
            r = item_r[1]
            q = pred[r]
            # u, s, vh = np.linalg.svd(tangent_spaces[q].T @ tangent_spaces[r])
            
            # R[r] = R[q] @ u @ vh
            # v[r] = v[q] + R[q] @ tangent_spaces[q].T @ (X[r] - X[q])
            # D[i, r] = np.linalg.norm(v[r]) # geo_dist[r]
            D[i, r] = D[i, q] + nnbrs_pairwise_dists[q, r]
            for j in nnbrs_idxs[r]:
                tmp_dist = dist[r] + nnbrs_pairwise_dists[j, r]
                if tmp_dist < dist[j]:
                    dist[j] = tmp_dist
                    pred[j] = r
                    if not j_used[j]:
                        heapq.heappush(P, (dist[j], j))
                        j_used[j] = True
    D = (D + D.T) / 2
    return D

def test():
    X, color = make_swiss_roll(n_samples=700, random_state=123)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.rainbow)
    plt.title('Swiss Roll in 3D')
    plt.show()

    x_embed = PTU(X, 2, n_neighbors=10)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(x_embed[:, 0], x_embed[:, 1], c=color, cmap=plt.cm.rainbow)
    plt.show()

def test1():
    X = np.random.randn(12).reshape((4, 3))
    print(X)
    T = _tangent_space(X, 0, [1, 2, 3], 3)
    print(T)
    print(T[:, 0] @ X[0])
    print(T[:, 1] @ X[0])

def test2():
    samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
    neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
    neigh.fit(samples)
    print(neigh.kneighbors([[1  , 0, 0]], 2, return_distance=False))

if __name__ == '__main__':
    test()
