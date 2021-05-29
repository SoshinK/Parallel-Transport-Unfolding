import heapq
import numpy as np
from tqdm import tqdm

def dijkstra(i, neighbors, ledges, n):
    P = []
    j_used = [False] * n

    D = np.zeros((n,))
    routes = [[] for _ in range(n)]
    nnbrs_idxs, nnbrs_pairwise_dists = neighbors, ledges

    pred = np.zeros((n), dtype=int)
    dist = np.ones((n)) * np.infty # ??
    dist[i] = 0
    

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
       
        D[r] = D[q] + nnbrs_pairwise_dists[q, r]
        for k in routes[q]:
            routes[r].append(k)    
        routes[r].append(q)
        for j in nnbrs_idxs[r]:
            tmp_dist = dist[r] + nnbrs_pairwise_dists[j, r]
            if tmp_dist < dist[j]:
                dist[j] = tmp_dist
                pred[j] = r
                if not j_used[j]:
                    heapq.heappush(P, (dist[j], j))
                    j_used[j] = True
    for k in range(len(routes)):
        routes[k].append(k)
    return D, routes