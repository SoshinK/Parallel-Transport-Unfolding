import numpy as np
import matplotlib.pyplot as plt

from utils import angle_between
from data_generator import sample_on_sphere
from dijkstra import dijkstra
from ptransport import PTU_dists

def experiment_geodesic():
    X, edges, nnbrs = sample_on_sphere(1000, random_state=123)
    # print("lol", edges)
    # edges -= np.eye(edges.shape[0])
    nnbrs_idxs = [np.nonzero(edges[i])[0] for i in range(edges.shape[0])]


    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.rainbow)

    for i in range(len(nnbrs_idxs)):
        for j in nnbrs_idxs[i]:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]])
    plt.title('Sphere')
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    # plt.zlim(-0.6, 0.6)
    plt.show()

    # i = 0
    # # print(nnbrs_idxs)
    # lengths, routes = dijkstra(i, nnbrs_idxs, nnbrs.kneighbors_graph(mode='distance').toarray(), X.shape[0])

    # ptu_dists = PTU_dists(X, 2)

    path_lengths = []
    dijkstra_errors = []#
    ptu_errors = []
    number_paths_that_length = []

    for i in [0, 50, 100, 150, 200, 250, 350, 400, 450]:
        # print(nnbrs_idxs)
        lengths, routes = dijkstra(i, nnbrs_idxs, nnbrs.kneighbors_graph(mode='distance').toarray(), X.shape[0])

        ptu_dists = PTU_dists(X, 2)


        for l, r in zip(lengths, routes):
            # print(path_lengths)
            true_geodesic = angle_between(X[i], X[int(r[-1])])
            if len(r) in path_lengths:
                idx = np.argwhere(np.array(path_lengths) == len(r))
                # print("keka", idx[0], len(r))
                dijkstra_errors[int(idx[0])] += np.abs(l - true_geodesic) / true_geodesic
                ptu_errors[int(idx[0])] += np.abs(ptu_dists[i, int(r[-1])] - true_geodesic) / true_geodesic
                number_paths_that_length[int(idx[0])] += 1
            else:
                path_lengths.append(len(r))
                dijkstra_errors.append(np.abs(l - true_geodesic) / true_geodesic)
                ptu_errors.append(np.abs(ptu_dists[i, int(r[-1])] - true_geodesic) / true_geodesic)
                number_paths_that_length.append(1)
        
        # print(lengths)
    dijkstra_errors_mean = np.array(dijkstra_errors) / np.array(number_paths_that_length)
    ptu_errors_mean = np.array(ptu_errors) / np.array(number_paths_that_length)

    path_lengths2 = path_lengths
    path_lengths, dijkstra_errors_mean = zip(*sorted(zip(path_lengths, dijkstra_errors_mean)))
    path_lengths, ptu_errors_mean = zip(*sorted(zip(path_lengths2, ptu_errors_mean)))

    plt.figsize((10, 8))
    plt.plot(path_lengths, dijkstra_errors_mean, label='dijkstra')
    plt.plot(path_lengths, ptu_errors_mean, c='r', label='PTU')
    plt.legend()
    plt.show()
    print(len(dijkstra_errors)) 
    print(len(number_paths_that_length))


if __name__ == '__main__':
    experiment_geodesic()