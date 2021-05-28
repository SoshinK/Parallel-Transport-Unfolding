import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import Isomap

from ptransport import PTU
from data_generator import s_roll_with_void, s_roll
from utils import overlay

def experiment_s_roll(n_samples, random_state, n_components , **params_for_ptu):
    X, color, gt = s_roll_with_void(n_samples, random_state=random_state, return_gt=True)
    
    x_ptu_solution = PTU(X, n_components, **params_for_ptu)
    
    isomap = Isomap(n_components=n_components)
    x_isomap_solution = isomap.fit_transform(X)
    
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(141, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.rainbow)
    plt.title('S Roll with void in 3D')

    ax = fig.add_subplot(142)
    ax.scatter(gt[:, 0], gt[:, 1], c=color, cmap=plt.cm.rainbow)
    plt.title('Unfolded S-Roll (ground Truth)')
    
    ax = fig.add_subplot(143)
    x_ptu_solution_rotated = overlay(x_ptu_solution, gt)
    # x_ptu_solution_rotated = x_ptu_solution
    sc = ax.scatter(
        x_ptu_solution_rotated[:, 0], 
        x_ptu_solution_rotated[:, 1], 
        c=np.sum((x_ptu_solution_rotated - gt)**2, axis=1),
        vmin=0.0,
        vmax=0.8,
        cmap=plt.cm.hot)
    plt.colorbar(sc)
    plt.title(f'PTU (mean err:{np.round(np.mean((x_ptu_solution_rotated - gt)**2), 4)})')


    ax = fig.add_subplot(144)
    x_isomap_solution_rotated = overlay(x_isomap_solution, gt)
    # x_isomap_solution_rotated = x_isomap_solution
    sc = ax.scatter(
        x_isomap_solution_rotated[:, 0], 
        x_isomap_solution_rotated[:, 1], 
        c=np.sum((x_isomap_solution_rotated - gt)**2, axis=1), 
        vmin=0.0,
        vmax=0.8,
        cmap=plt.cm.hot)
    plt.colorbar(sc)
    plt.title(f'Isomap (mean err:{np.round(np.mean((x_isomap_solution_rotated - gt)**2), 2)})')

    plt.show()


if __name__ == '__main__':
    experiment_s_roll(1500, 123, 2, n_neighbors=10)
    
