import numpy as np


class RLS:
    def __init__(self, n, mu=0.1, eps=0.001, w="random"):
        self.w = self.init_weights(w, n)
        self.n = n
        self.w_history = False
        self.mu = mu
        self.eps = eps
        self.R = 1/self.eps * np.identity(n)

    def init_weights(self, w, n=-1):
        if n == -1:
            n = self.n

        if isinstance(w, str):
            if w == "random":
                w = np.random.normal(0, 0.5, n)
            elif w == "zeros":
                w = np.zeros(n)
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

    def learning_rule(self, e, x):
        R1 = self.R @ (x[:, None] * x[None, :]) @ self.R
        R2 = self.mu + np.dot(np.dot(x, self.R), x.T)
        self.R = 1 / self.mu * (self.R - R1/R2)
        return np.dot(self.R, x.T)*e

    def adapt(self, d, x):
        y  = self.predict(x)
        e = d - y
        self.w += self.learning_rule(e, x)

    def predict(self, x):
        return np.dot(self.w, x)

