import heapq
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def _tangent_space(X, i, neighbours_idxs, d):
    geodesic_neighborhood = X[neighbours_idxs] - X[i]
    # print("!", geodesic_neighborhood.shape)
    u, _, _ = np.linalg.svd(geodesic_neighborhood.T)
    # print(">>", u[:, :d])
    return u[:, :d]


def _compute_tangent_spaces(X, nnbrs, K, d):
    T = []
    for i in tqdm(range(X.shape[0])):
        neighbours_idxs = nnbrs.kneighbors([X[i]], n_neighbors = K + 1,  return_distance=False)[0][1:]
        T.append(_tangent_space(X, i, neighbours_idxs, d))
    return T

def _get_neigh_idxs(i, nnbrs, n_neighbors, make_graph_how):
    if make_graph_how == 'neighbors':
        # return nnbrs.kneighbors([x], n_neighbors + 1, return_distance=False)[0][1:]
        idxs = nnbrs.kneighbors_graph().toarray()[i]
        idxs = np.argwhere(idxs == 1)[:, 0]
        return idxs[idxs != i]
    if make_graph_how == 'radius':
        # _, idxs = nnbrs.radius_neighbors([x], n_neighbors + 1, return_distance=True, sort_results=True)
        # return idxs[0][1:]
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
    print("fitted")
    tangent_spaces = _compute_tangent_spaces(X, nnbrs, K, n_components)
    print("tangent yes")
    P = []
    j_used = [False] * n

    R = np.zeros((n, n_components, n_components))
    v = np.zeros((n, n_components))
    D = np.zeros((n, n))
    # pred = np.zeros((n), dtype=int)
    # dist = np.zeros((n)) # ??

    for i in tqdm(range(n)):
        pred = np.zeros((n), dtype=int)
        dist = np.ones((n)) * np.infty # ??
        
        R[i] = np.eye(n_components)
        v[i] = np.zeros((n_components))
        
        adjacent_idxs = _get_neigh_idxs(i, nnbrs, n_neighbors, make_graph_how)

        for j in adjacent_idxs:
            pred[j] = int(i)
            dist[j] = np.linalg.norm(X[j] - X[i])
            heapq.heappush(P, (dist[j], j))
            j_used[j] = True
        
        while P: # while P is not empty
            item_r = heapq.heappop(P)
            j_used[item_r[1]] = False
            # print("??", item_r)
            r = item_r[1]
            # x_r = X[r]
            q = pred[r]
            # print(tangent_spaces[q].shape, tangent_spaces[r].shape)
            u, s, vh = np.linalg.svd(tangent_spaces[q].T @ tangent_spaces[r])
            
            R[r] = R[q] @ u @ vh
            v[r] = v[q] + R[q] @ tangent_spaces[q].T @ (X[r] - X[q])
            # print(r)
            D[i, r] = np.linalg.norm(v[r]) # geo_dist[r]
            
            adjacent_to_r_idxs = _get_neigh_idxs(r, nnbrs, n_neighbors, make_graph_how)
            print(adjacent_to_r_idxs)
            for j in adjacent_to_r_idxs:
                tmp_dist = dist[r] + np.linalg.norm(X[j] - X[r])
                if tmp_dist < dist[j]:
                    dist[j] = tmp_dist
                    pred[j] = r
                    print("lol", j)
                    if not j_used[j]:
                        heapq.heappush(P, (dist[j], j))
                        j_used[j] = True
    D = (D + D.T) / 2
    return D



def dijkstra(neighbors, ledges, n):
    P = []
    # R = np.zeros((n, n_components, n_components))
    # v = np.zeros((n, n_components))
    D = np.zeros((n, n))
    # pred = np.zeros((n), dtype=int)
    # dist = np.zeros((n)) # ??

    for i in tqdm(range(n)):
        pred = np.zeros((n), dtype=int)
        dist = np.ones((n)) * np.infty # ??
        # R[i] = np.eye(n_components)
        # v[i] = np.zeros((n_components))
        
        # adjacent_idxs = _get_neigh_idxs(X[i], nnbrs, n_neighbors, make_graph_how)

        for j in neighbors[i]:
            pred[j] = int(i)
            dist[j] = ledges[i, j]
            heapq.heappush(P, (dist[j], j))
        print(">>", i, pred, dist)
        while P: # while P is not empty
            print("heap:", P)
            item_r = heapq.heappop(P)
            print("??", item_r)
            r = item_r[1]
            # x_r = X[r]
            q = pred[r]
            # # print(tangent_spaces[q].shape, tangent_spaces[r].shape)
            # u, s, vh = np.linalg.svd(tangent_spaces[q].T @ tangent_spaces[r])
            # R[r] = R[q] @ u @ vh
            # v[r] = v[q] + R[q] @ tangent_spaces[q].T @ (X[r] - X[q])
            # # print(r)
            D[i, r] = D[i, q] + ledges[r, q] # geo_dist[r]
            print("lol ", D)
            for j in neighbors[r]:
                tmp_dist = dist[r] + ledges[j, r]#np.linalg.norm(X[j] - X[r])
                if tmp_dist < dist[j]:
                    dist[j] = tmp_dist
                    pred[j] = r
                    if not j in P:
                        heapq.heappush(P, (dist[j], j))
    # D = (D + D.T) / 2
    return D


def test():
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
    test2()
