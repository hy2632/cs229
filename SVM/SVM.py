import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from tqdm.notebook import tqdm


class SVM():
    """ 
    SVM binary classifier, optimizing with the dual lagrangian program, 
    trained on batch. Uses the SMO algorithm.

    [CS229 - Kernel Methods and SVM](http://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes3.pdf)

    ## Input:
        X: (N, d)
        Y: (N,)
        n = batch_size: int
        C: l1 regularization term. alpha_i in [0, C]

    ## Intermediate parameters:
        x: (n, d), random batch of X
        y: (n)
        M: (n, n), M[i,j] = y_iy_j<x_i,x_j>

    ## Learned parameters:
        alpha: (n,)
        w: (d,)
        b: int

    """

    def __init__(self, X, Y, batch_size=64, C=5):
        # X: (N,d), Y: (N,), x: (n,d), y:(n,)
        # Fixed values
        self.N, self.d = X.shape
        self.n = batch_size
        self.C = C
        self.X = X
        self.Y = Y
        self.x = self.X[:self.n]
        self.y = self.Y[:self.n]
        self.M = np.outer(self.y,self.y) * np.dot(self.x, self.x.T)

        
        # Learned parameters
        self.alpha = np.ones(self.n)
        self.w = np.zeros(self.d)
        self.b = 0
    
    def shuffle_batch(self):
        batch_indices = np.random.choice(np.arange(self.N), self.n)
        self.x = self.X[batch_indices]
        self.y = self.Y[batch_indices]
        self.M = np.outer(self.y,self.y) * np.dot(self.x, self.x.T)

    def update_alpha(self, random_idx1, random_idx2):
        Zeta = -np.sum(self.alpha*self.y) + (self.alpha*self.y)[random_idx1] + (self.alpha*self.y)[random_idx2]
        self.alpha[random_idx1] = (Zeta - self.alpha[random_idx2]*self.y[random_idx2])*self.y[random_idx1]

    def dual_obj(self, alpha):
        return np.sum(alpha) - np.sum(0.5 * self.M * np.outer(alpha,alpha))
    
    def optimize(self, iterations = 100000, tol=1e-2):
        prev_val = self.dual_obj(self.alpha)
        # The first shuffling happens at time 0
        shuffle_timepoint=0


        for i in tqdm(range(iterations)):
            self.w = np.dot(self.alpha*self.y, self.x)
            self.b = (np.max(np.dot(self.x, self.w)[self.y==-1]) + np.min(np.dot(self.x, self.w)[self.y==1])) * -0.5

            # Select 2 alphas randomly
            random_idx1, random_idx2 = np.random.choice(np.arange(0, self.n), 2)

            # The (quadratic w.r.t a2) function that scipy.optimize.minimize takes
            def optimizeWRTa2(a2):
                self.alpha[random_idx2] = a2
                self.update_alpha(random_idx1, random_idx2)
                return -self.dual_obj(self.alpha)

            # Solve optimization w.r.t a2
            a2 = self.alpha[random_idx2]
            res = minimize(optimizeWRTa2, a2, bounds=[(0,self.C)])
            a2 = res.x

            # Update a2 in alpha
            self.alpha[random_idx2] = a2
            self.update_alpha(random_idx1, random_idx2)

            # Check convergence
            if (i%100 == 1):
                if np.abs(self.dual_obj(self.alpha) - prev_val) < tol:
                    if i - shuffle_timepoint < 200:
                        print("Converges quickly after shuffling. End early.")
                        return
                    else:
                        print(f"Optimized on one batch: {self.dual_obj(self.alpha) - prev_val}, shuffling... \n")
                        self.shuffle_batch()
                        # Record the time of shuffling. If next shuffling appears soon,
                        # just end the training.
                        shuffle_timepoint = i
                        continue
                else:
                    if (i%1000 == 1):
                        print(self.dual_obj(self.alpha) - prev_val)
                    prev_val = self.dual_obj(self.alpha)

