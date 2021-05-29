import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from  scipy import interpolate
import matplotlib.pyplot as plt

from utils import angle_between
from data_generator import sample_on_sphere
from dijkstra import dijkstra
from ptransport import PTU_dists
import warnings

from pydiffmap import diffusion_map as dm

def get_dm_dists(X):
    embedd_dim = 2
    neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
    mydmap = dm.DiffusionMap.from_sklearn(n_evecs=embedd_dim, k=200, epsilon='bgh', 
                                            alpha=1.0, neighbor_params=neighbor_params)
    x_dmap_solution = mydmap.fit_transform(X)
    dist_matrix = np.empty((X.shape[0], X.shape[0]), dtype=np.float_)
    for i, x in enumerate(x_dmap_solution):
        for j, y in enumerate(x_dmap_solution):
            if i < j:
                continue
            dist_matrix[i, j] = scipy.linalg.norm(x-y)
            dist_matrix[j, i] = scipy.linalg.norm(x-y)
    return dist_matrix
    

warnings.filterwarnings('error')
def experiment_geodesic():
    X, edges, nnbrs = sample_on_sphere(1000, random_state=123)
    # print("lol", edges)
    # edges -= np.eye(edges.shape[0])
    nnbrs_idxs = [np.nonzero(edges[i])[0] for i in range(edges.shape[0])]


    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.rainbow)

    for i in range(len(nnbrs_idxs)):
        for j in nnbrs_idxs[i]:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]])
    plt.title('Sphere')
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    # plt.zlim(-0.6, 0.6)
    # plt.show()

    # i = 0
    # # print(nnbrs_idxs)
    # lengths, routes = dijkstra(i, nnbrs_idxs, nnbrs.kneighbors_graph(mode='distance').toarray(), X.shape[0])

    # ptu_dists = PTU_dists(X, 2)

    path_lengths = []
    dijkstra_errors = []#
    ptu_errors = []
    dm_errors = []
    number_paths_that_length = []

    for i in [0, 50, 150 , 250, 350, 450]:
        # print(nnbrs_idxs)
        lengths, routes = dijkstra(i, nnbrs_idxs, nnbrs.kneighbors_graph(mode='distance').toarray(), X.shape[0])

        ptu_dists = PTU_dists(X, 2)
        # dm_disits = get_dm_dists(X)
        # np.seterr(all='raise')
        eps = 1e-7
        for l, r in zip(lengths, routes):
            # print(path_lengths)
            true_geodesic = angle_between(X[i], X[int(r[-1])]) + eps
            if len(r) in path_lengths:
                idx = np.argwhere(np.array(path_lengths) == len(r))
                # print("keka", idx[0], len(r))
                dijkstra_errors[int(idx[0])] += np.abs(l - true_geodesic) / true_geodesic
                ptu_errors[int(idx[0])] += np.abs(ptu_dists[i, int(r[-1])] - true_geodesic) / true_geodesic
                # # dm_errors[int(idx[0])] += np.abs(dm_disits[i, int(r[-1])] - true_geodesic) / true_geodesic
                number_paths_that_length[int(idx[0])] += 1
            else:
                try:
                    path_lengths.append(len(r))
                    dijkstra_errors.append(np.abs(l - true_geodesic) / true_geodesic)
                    ptu_errors.append(np.abs(ptu_dists[i, int(r[-1])] - true_geodesic) / true_geodesic)
                    # # dm_errors.append(np.abs(dm_disits[i, int(r[-1])] - true_geodesic) / true_geodesic)
                    number_paths_that_length.append(1)
                except Warning:
                    print(i, r[-1], true_geodesic, ptu_dists[i, int(r[-1])])
                    exit()
        
        # print(lengths)
    dijkstra_errors_mean = np.array(dijkstra_errors) / np.array(number_paths_that_length)
    ptu_errors_mean = np.array(ptu_errors) / np.array(number_paths_that_length)
    # # dm_errors_mean = np.array(dm_errors) / np.array(number_paths_that_length)

    path_lengths2 = path_lengths
    path_lengths3 = path_lengths
    path_lengths, dijkstra_errors_mean = zip(*sorted(zip(path_lengths, dijkstra_errors_mean)))
    path_lengths, ptu_errors_mean = zip(*sorted(zip(path_lengths2, ptu_errors_mean)))
    # # path_lengths, dm_errors_mean = zip(*sorted(zip(path_lengths3, dm_errors_mean)))

    # plt.figure(figsize=(10, 8))
    x_smooth = np.linspace(np.min(path_lengths), np.max(path_lengths), 20)
    sigma = 5
    # x_g1d = gaussian_filter1d(path_lengths, sigma)
    # y_g1d = gaussian_filter1d(dijkstra_errors_mean, sigma)
    spl = interpolate.UnivariateSpline(path_lengths, dijkstra_errors_mean)
    # spl2 = interpolate.UnivariateSpline(path_lengths, dm_errors_mean)
    ax2 = fig.add_subplot(122)
    ax2.plot(path_lengths, dijkstra_errors_mean, label='dijkstra')
    # ax2.plot(path_lengths, dm_errors_mean, label='diffusion distance')
    ax2.plot(path_lengths, ptu_errors_mean, c='r', label='PTU')
    ax2.plot()
    ax2.legend()
    ax2.set_xlabel('Number of graph edges in geodesic')
    ax2.set_ylabel('Mean error')
    plt.show()
    # print(len(dijkstra_errors)) 
    # print(len(number_paths_that_length))


if __name__ == '__main__':
    experiment_geodesic()