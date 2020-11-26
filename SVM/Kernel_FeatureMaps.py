import numpy as np


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


def orthgonalize(V):
    N = V.shape[0]
    d = V.shape[1]
    turns = int(N / d)
    remainder = N % d

    V_ = np.zeros_like(V)

    for i in range(turns):
        v = gram_schmidt_columns(V[i * d:(i + 1) * d, :].T).T
        V_[i * d:(i + 1) * d, :] = v
    if remainder != 0:
        V_[turns * d:, :] = gram_schmidt_columns(V[turns * d:, :].T).T

    return V_


# Generate orthogonal normal weights (w1, ..., wm)
def generateGMatrix(m, d) -> np.array:
    G = np.random.normal(0, 1, (m, d))
    # Renormalize
    norms = np.linalg.norm(G, axis=1).reshape([m, 1])
    return orthgonalize(G) * norms


# Softmax trignometric feature map Φ(x)
def baseline_SM(x, G):
    """Calculate the result of softmax trigonometrix random feature mapping

        Parameters
        ----------
        x: array, dimension = n*d
            Input to the baseline mapping
            Required to be of norm 1 for i=1,...n
        
        G: matrix, dimension = m*d
            The matrix in the baseline random feature mapping

    """
    m = G.shape[0]
    left = np.cos(np.dot(x, G.T).astype(np.float32))
    right = np.sin(np.dot(x, G.T).astype(np.float32))
    return np.exp(0.5) * ((1 / m)**0.5) * np.hstack([left, right])


class Kernel(object):
    """Kernels.

        Parameters
        ---------
        x, y: array of (n_x, d), (n_y, d)

        Return
        ---------
        K(x,y): kernel matrix of (n_x, n_y), K_ij = K(x_i, y_j)

    """
    @staticmethod
    def Linear(x, y):
        return np.dot(x, y.T)

    @staticmethod
    def Softmax(x, y):
        return np.exp(np.dot(x, y.T))


class FeatureMap(object):
    """Feature mapping
    
        Parameters
        ---------
        x: array of (n, d)

        Return
        ---------
        Φ(x): array of (n, m), where m is usually a higher dimensionality. Here we set m = 2*n

    """
    @staticmethod
    def Linear(x):
        return x

    @staticmethod
    def Softmax_Trigonometric(x):
        n, d = x.shape
        # Increase dimensionality to 2x
        G = generateGMatrix(2 * d, d)
        phi_x = baseline_SM(x, G)
        return phi_x
