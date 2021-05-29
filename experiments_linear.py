import numpy as np
import matplotlib.pyplot as plt

from pydiffmap import diffusion_map as dm
from sklearn.manifold import Isomap

from ptransport import PTU
from data_generator import plane
from utils import overlay

def experiment_plane(n_samples, random_state, n_components , **params_for_ptu):
    X, color, gt = plane(n_samples, random_state=random_state, return_gt=True)
    
    # PTU solution
    x_ptu_solution = PTU(X, n_components, **params_for_ptu)
    # Isomap solution
    isomap = Isomap(n_components=n_components)
    x_isomap_solution = isomap.fit_transform(X)
    # Diffusion map solution
    neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
    mydmap = dm.DiffusionMap.from_sklearn(n_evecs=2, k=200, epsilon='bgh', 
                                          alpha=1.0, neighbor_params=neighbor_params)
    x_dmap_solution = mydmap.fit_transform(X)
    
    fig = plt.figure(figsize=(23,5))
    ax = fig.add_subplot(151, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.rainbow)
    plt.title('Plane in 3D')

    ax = fig.add_subplot(152)
    ax.scatter(gt[:, 0], gt[:, 1], c=color, cmap=plt.cm.rainbow)
    plt.title('Unfolded Plane (ground Truth)')
    
    ax = fig.add_subplot(153)
    x_ptu_solution_rotated = overlay(x_ptu_solution, gt)
    # x_ptu_solution_rotated = x_ptu_solution
    sc = ax.scatter(
        x_ptu_solution_rotated[:, 0], 
        x_ptu_solution_rotated[:, 1], 
        c=np.sum((x_ptu_solution_rotated - gt)**2, axis=1),
        # vmin=0.0,
        # vmax=0.8,
        cmap=plt.cm.hot)
    plt.colorbar(sc)
    plt.title(f'PTU (mean err:{np.round(np.mean((x_ptu_solution_rotated - gt)**2), 7)})')


    ax = fig.add_subplot(154)
    x_isomap_solution_rotated = overlay(x_isomap_solution, gt)
    # x_isomap_solution_rotated = x_isomap_solution
    sc = ax.scatter(
        x_isomap_solution_rotated[:, 0], 
        x_isomap_solution_rotated[:, 1], 
        c=np.sum((x_isomap_solution_rotated - gt)**2, axis=1), 
        # vmin=0.0,
        # vmax=0.8,
        cmap=plt.cm.hot)
    plt.colorbar(sc)
    plt.title(f'Isomap (mean err:{np.round(np.mean((x_isomap_solution_rotated - gt)**2), 5)})')

    ax = fig.add_subplot(155)
    x_dmap_solution_rotated = overlay(x_dmap_solution, gt)
    # x_dmap_solution_rotated = x_dmap_solution
    sc = ax.scatter(
        x_dmap_solution_rotated[:, 0], 
        x_dmap_solution_rotated[:, 1], 
        c=np.sum((x_dmap_solution_rotated - gt)**2, axis=1), 
        vmin=0.0,
        vmax=0.8,
        cmap=plt.cm.hot)
    plt.colorbar(sc)
    plt.title(f'Diffusion map (mean err:{np.round(np.mean((x_dmap_solution_rotated - gt)**2), 2)})')

    plt.tight_layout()
    plt.subplots_adjust()
    plt.show()


if __name__ == '__main__':
    experiment_plane(1000, 123, 2, n_neighbors=10)