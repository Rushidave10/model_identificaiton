import control as ct
import matplotlib.pyplot as plt
import numpy as np

class LCfilter:
    def __init__(self, L, C, R, Omega):
        self.L = L
        self.C = C
        self.R = R
        self.Omega = Omega

        A = np.array([[-self.R / self.L, -self.Omega, -1 / (3 * self.L), 0],
                      [self.Omega, -self.R / self.L, 0, -1 / (3 * self.L)],
                      [1 / (3 * self.C), 0, 0, -self.Omega],
                      [0, 1 / (3 * self.C), self.Omega, 0],
                      ])

        B = np.array([[1 / (3 * self.L), 0],
                      [0, 1 / (3 * self.L)],
                      [0, 0],
                      [0, 0],
                      ])
        C = np.diag(np.ones(4))

        D = np.zeros(8).reshape((4, 2))
        E = np.array([[0, 0],
                      [0, 0],
                      [-1/3*self.C, 0],
                      [0, -1/3*self.C]])

        self.sys = ct.ss(A, B, C, D,)

    def forced_response(self, T, plot=False, return_x=False, inputs=np.ones(100)):
        result = ct.forced_response(sys=self.sys,
                                    T=T,
                                    U=inputs
                                    )
        if plot:
            plt.plot(result.time, result.outputs)
            plt.show()
        return result
