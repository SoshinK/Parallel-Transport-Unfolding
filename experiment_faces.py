import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.manifold import Isomap
from ptransport import PTU

def experiment_faces():
    data_faces = fetch_olivetti_faces(shuffle=True).data[::15]
    data_embed = PTU(data_faces, 3, n_neighbors=8)
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(121)
    ax.plot(data_embed[:, 0], data_embed[:, 1], 'o')
    plt.title("Olivetti faces (PTU)")
    for i in range(data_faces.shape[0]):
        ax.text(data_embed[i, 0], data_embed[i, 1], i)

    isomap = Isomap(3)
    isomap_emd = isomap.fit_transform(data_faces)
    
    ax = fig.add_subplot(122)
    ax.plot(isomap_emd[:, 0], isomap_emd[:, 1], 'o')
    plt.title("Olivetti faces (Isomap)")
    for i in range(data_faces.shape[0]):
        ax.text(isomap_emd[i, 0], isomap_emd[i, 1], i)

    plt.show()

def show_faces():
    data_faces = fetch_olivetti_faces(shuffle=True).data[::15]
    for i, f in enumerate(data_faces):
        plt.imshow(f.reshape((64, 64)), cmap='gray')
        plt.title(i)
        plt.show()

if __name__ == '__main__':
    show_faces()