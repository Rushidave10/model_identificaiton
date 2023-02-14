import numpy as np


def compute_delta(R, x, e):
    return np.dot(R, x.T) * e


class RLS:
    def __init__(self, n, p=1, mu=0.1, eps=0.001, w="random"):
        """
        p  : Number of outputs
        n  : Number of filter coefficients
        mu : Forgetting factor (0 < mu <= 1)
        eps: Positive Constant
        w  : Initialization of weights
        """
        self.R = None
        self.w = self.init_weights(w, p, n)
        self.n = n
        self.p = p
        self.mu = mu
        self.eps = eps
        if p == 1:
            self.R = self.init_gain(p, n)
        else:
            self.delta_w = np.zeros((n, p)).T
            self.R = self.init_gain(p, n)

    def init_weights(self, w, p, n=-1):
        if n == -1:
            n = self.n

        if isinstance(w, str):
            if w == "random":
                if p == 1:
                    w = np.random.normal(0, 0.5, n)
                else:
                    w = np.random.normal(0, 0.5, (n, p))
            elif w == "zeros":
                if p == 1:
                    w = np.zeros(n)
                else:
                    w = np.zeros((n, p))
            else:
                raise ValueError("Choose either (random, zeros) string type")

        elif len(w) == n:
            try:
                w = np.array(w, dtype="float64")
            except:
                raise ValueError("Matrix size not equal to no. of filter co-efficents")
        else:
            raise ValueError("Chose either a string or a Matrix")
        return w

    def init_gain(self, p, n):
        if p == 1:
            self.R = 1 / self.eps * np.identity(n)

        else:
            self.R = self.identity_matrix_stack(n, p)

        return self.R

    def identity_matrix_stack(self, n, p):
        single_id_matrix = 1 / self.eps * np.identity(n)
        stacket_id_matrix = np.stack([single_id_matrix] * p)
        return stacket_id_matrix

    def learning_rule(self, e, x):
        """
        p = Number of output
        """
        if self.p == 1:
            R1 = self.R @ (x[:, None] * x[None, :]) @ self.R
            R2 = self.mu + np.dot(np.dot(x, self.R), x.T)
            self.R = 1 / self.mu * (self.R - R1 / R2)
            return np.dot(self.R, x.T) * e
        else:
            for idx in range(self.p):
                R1 = self.R[idx] @ (x[:, None] * x[None, :]) @ self.R[idx]
                R2 = self.mu + np.dot(np.dot(x, self.R[idx]), x.T)
                self.R[idx] = 1 / self.mu * (self.R[idx] - R1 / R2)

                self.delta_w[idx, :] = compute_delta(self.R[idx], x, e[idx])
            return self.delta_w

    def adapt(self, d, x):
        y = self.predict(x, self.p)
        e = d - y
        self.w += self.learning_rule(e, x).T

    def predict(self, x, p=1):
        if p == 1:
            return np.dot(self.w, x)
        else:
            return np.dot(self.w.T, x)
