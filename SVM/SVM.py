import numpy as np
from scipy.optimize import minimize
from tqdm.notebook import tqdm
from Kernel_FeatureMaps import *;

class SVM(object):
    """ 
    ## Author:
        Hua Yao (hy2632@columbia.edu)

    ## Description:
        SVM binary classifier, optimizing with the dual lagrangian program, trained on a sample batch. Uses the SMO algorithm.
        Normalizes input X to adapt to kernelized version.
        Regularization parameter C set to be `np.inf` for easy closed form solution of b.

    ## Reference:
        [CS229 - Kernel Methods and SVM](http://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes3.pdf)

    ---

        Parameters:
        ---------
        X: (N, d)
        Y: (N,)
        kernel: kernel used, linear/softmax
        featuremap: feature mapping corresponding to the kernel used
        batch_size: int, also denoted as n
        C: l1 regularization term for soft margin. alpha_i in [0, C]. Set as np.inf (no regularization), because the form of b is nasty under regularization.
        tol = 1e-6: tolerance, deciding when to end training

        Intermediate parameters:
        ---------
        x: (n, d), random batch of X
        y: (n)
        phi_x: (n, m), feature map(s) of x
        M: (n, n), M[i,j] = y_iy_j * K(x_i,x_j), hadamard product of y^Ty and K

        Learned parameters:
        ---------
        alpha: (n,)
        w: (d,)
        b: int

    """
    def __init__(self,
                 X,
                 Y,
                 kernel=Kernel.Linear,
                 featuremap=FeatureMap.Linear,
                 batch_size=64,
                 C=np.inf,
                 tol=1e-6):
        # C set as np.inf here -- no regularization

        # X: (N,d), Y: (N,), x: (n,d), y:(n,)
        # Fixed values
        self.N, self.d = X.shape
        # Normalize data
        self.X = X
        self.X = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)
        self.Y = Y
        self.kernel = kernel
        self.featuremap = featuremap
        self.n = batch_size
        self.C = C
        self.tol = tol

        batch_indices = np.random.choice(np.arange(self.N), self.n)
        self.x = self.X[batch_indices]
        self.y = self.Y[batch_indices]
        self.phi_x = self.featuremap(self.x)
        self.M = np.outer(self.y, self.y) * self.kernel(self.x, self.x)

        # Learned parameters
        self.alpha = np.ones(self.n)
        self.w = np.zeros(self.d)
        self.b = 0

    def update_alpha(self, random_idx1, random_idx2):
        Zeta = -np.sum(self.alpha * self.y) + (
            self.alpha * self.y)[random_idx1] + (self.alpha *
                                                 self.y)[random_idx2]
        self.alpha[random_idx1] = (Zeta - self.alpha[random_idx2] *
                                   self.y[random_idx2]) * self.y[random_idx1]

    def dual_obj(self, alpha):
        return np.sum(alpha) - np.sum(0.5 * self.M * np.outer(alpha, alpha))

    def fit(self, iterations=200000):
        prev_val = self.alpha.copy()

        for i in tqdm(range(iterations)):
            # Select 2 alphas randomly
            random_idx1, random_idx2 = np.random.choice(
                np.arange(0, self.n), 2)

            # The (quadratic w.r.t a2) function that scipy.optimize.minimize takes
            def optimizeWRTa2(a2):
                self.alpha[random_idx2] = a2
                self.update_alpha(random_idx1, random_idx2)
                return -self.dual_obj(self.alpha)

            # Solve optimization w.r.t a2
            a2 = self.alpha[random_idx2]
            res = minimize(optimizeWRTa2, a2, bounds=[(0, self.C)])
            a2 = res.x

            # Update a2
            self.alpha[random_idx2] = a2
            self.update_alpha(random_idx1, random_idx2)

            # Check convergence
            if (i % 5 == 1):
                if np.sum(np.abs(self.alpha - prev_val)) < self.tol:
                    print(
                        f">> Optimized on the batch, step {i}. 5 steps Δalpha:{np.sum(np.abs(self.alpha - prev_val))}"
                    )
                    return
                else:
                    if (i % 5000 == 1):
                        print(
                            f">> Optimizing, step {i}. Δalpha:{np.sum(np.abs(self.alpha - prev_val))}"
                        )
                    prev_val = self.alpha.copy()

        # Retrieve w and b
        self.w = np.dot(self.alpha * self.y, self.phi_x)
        # The form of b changes when there exists Regularization C. So we simply cancel C here.
        # Look at P25 of CS229 - Note 3.
        # Form of b under regularization depends on alpha. (http://cs229.stanford.edu/materials/smo.pdf)
        # [Bias Term b in SVMs Again](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2004-11.pdf) gives a general representation
        self.b = (np.max(np.dot(self.phi_x, self.w)[self.y == -1]) +
                  np.min(np.dot(self.phi_x, self.w)[self.y == 1])) * -0.5

    def predict(self, X_val):
        X_val_normed = X_val / np.linalg.norm(X_val, axis=1, keepdims=True)
        return np.sign(
            np.dot(self.kernel(X_val_normed, self.x), self.alpha * self.y) +
            self.b)

    def score(self, X_val, y_val):
        prediction = self.predict(X_val)
        return np.mean(prediction == y_val)
