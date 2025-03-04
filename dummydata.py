import h5py
import scipy
import sklearn
import sklearn.datasets
import sklearn.linear_model

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def generate_dummy_data(Type,Visualize=0):
    if (Type == 'circular'):
        np.random.seed(1)
        train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
        np.random.seed(2)
        test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
        if(Visualize==1):
            plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        test_X = test_X.T
        test_Y = test_Y.reshape((1, test_Y.shape[0]))
        return train_X, train_Y, test_X, test_Y
    
    
    if (Type == 'plannar'):
        np.random.seed(1)
        m = 400  # number of examples
        N = int(m / 2)  # number of points per class
        D = 2  # dimensionality
        X = np.zeros((m, D))  # data matrix where each row is a single example
        Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
        a = 4  # maximum ray of the flower

        for j in range(2):
            ix = range(N * j, N * (j + 1))
            t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
            r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Y[ix] = j

        X = X.T
        Y = Y.T

        return X, Y
    
    if(Type == 'extra'):
        N = 200
        noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
        noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
        blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
        gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,n_classes=2, shuffle=True, random_state=None)
        no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

        return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
    if(Type == '2D'):
        data = scipy.io.loadmat('datasets/data.mat')
        train_X = data['X'].T
        train_Y = data['y'].T
        test_X = data['Xval'].T
        test_Y = data['yval'].T
        if (Visualize == 1):
            plt.scatter(train_X[0, :], train_X[1, :], c=train_Y[0,:], s=40,cmap=plt.cm.Spectral);

        return train_X, train_Y, test_X, test_Y
