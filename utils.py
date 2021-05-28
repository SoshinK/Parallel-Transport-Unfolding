import numpy as np

def get_closest_to(source, target):
    '''
    bla bla


    target.T = M * source.T 
    or 
    target = source * T

    T = inv(source.T * source) * source.T * y - pseudoinverse

    Parameters
    ----------

    target : np.ndarray
        Shape [n_samples, n_features]
    source : np.ndarray
        Shape [n_samples, n_features]
    '''
    return np.linalg.inv(source.T @ source) @ source.T @ target

def overlay(source, target):
    T = get_closest_to(source, target)
    return source @ T

from data_generator import s_roll_with_void
import matplotlib.pyplot as plt
def test():
    _, color, gt = s_roll_with_void(1000, return_gt=True)

    phi = np.pi / 180 * 45

    T = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

    rotated = gt @ T

    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(141)
    ax1.scatter(gt[:, 0], gt[:, 1], c=color, cmap=plt.cm.rainbow)
    ax2 = fig.add_subplot(142)
    ax2.scatter(rotated[:, 0], rotated[:, 1], c=color, cmap=plt.cm.rainbow)

    pred = overlay(rotated, gt)
    ax3 = fig.add_subplot(143)
    ax3.scatter(pred[:, 0], pred[:, 1], c=color, cmap=plt.cm.rainbow)
    ax3 = fig.add_subplot(144)
    sc= ax3.scatter(pred[:, 0], pred[:, 1], c=np.sum((pred-gt) ** 2, axis=1).astype(np.float32),vmin=0, vmax=10, cmap=plt.cm.rainbow)
    plt.colorbar(sc,)
    plt.show()


if __name__=='__main__':
    test()