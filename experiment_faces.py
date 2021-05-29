import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from ptransport import PTU

def experiment_faces():
    data_faces = fetch_olivetti_faces(shuffle=True).data[::15]
    data_embed = PTU(data_faces, 2)
    plt.plot(data_embed[:, 0], data_embed[:, 1], 'o')
    plt.title("Olivetti faces")
    plt.show()

if __name__ == '__main__':
    experiment_faces()