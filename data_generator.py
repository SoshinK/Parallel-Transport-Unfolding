import numpy as np
import matplotlib.pyplot as plt

def s_roll(n_samples, noise = 0.0, random_state=123, return_gt=False):
    if return_gt and noise > 0:
        raise Warning("return_gt is set to `True` but noise is not zero. Ground truth is not accurate")
    
    np.random.seed(random_state)
    t = 3 * np.pi * (np.random.rand(1, n_samples) - 0.5)
    y = 2.0 * (np.random.rand(1, n_samples) - 0.5)
    

    x = np.sin(t)
    z = np.sign(t) * (np.cos(t) - 1)

    X = np.concatenate((x, y, z))
    X += noise * np.random.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)    
    if return_gt:
        return X, t, np.array([t, y[0]]).T
    else:
        return X, t

def s_roll_with_void(n_samples, noise = 0.0, random_state=123, return_gt=False):
    if return_gt and noise > 0:
        raise Warning("return_gt is set to `True` but noise is not zero. Ground truth is not accurate")
    
    np.random.seed(random_state)
    t = 3 * np.pi * (np.random.rand(1, n_samples) - 0.5)
    y = 2.0 * (np.random.rand(1, n_samples) - 0.5)

    gt = np.array([t[0], y[0]]).T
    mask1 = gt[:, 0] < 2 
    mask2 = gt[:, 0] > -2
    mask3 = gt[:, 1] < 0.25 
    mask4 = gt[:, 1] > -0.25
    mask = mask1 & mask2 & mask3 & mask4

    x = np.sin(t)
    z = np.sign(t) * (np.cos(t) - 1)

    X = np.concatenate((x, y, z))
    X += noise * np.random.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)    
    if return_gt:
        return X[~mask], t[~mask], gt[~mask]
    else:
        return X[~mask], t[~mask]


def test_s():
    X, color, gt = s_roll_with_void(4000, return_gt=True)

    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.rainbow)
    plt.title('S Roll in 3D')

    ax2 = fig.add_subplot(122)
    ax2.scatter(gt[:, 0], gt[:, 1], c=color, cmap=plt.cm.rainbow)

    plt.show()

if __name__ == '__main__':
    test_s()