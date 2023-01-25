import argparse
import control as ct
import matplotlib.pyplot as plt
import numpy as np

"""
A Spring-mass damped system with mass, spring constant and damping constant as parameters.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--mass", type=float, default=250.0, help="Mass of the system")
parser.add_argument("--spring_constant", type=float, default=40.0, help="spring constant of the system")
parser.add_argument("--damping_constant", type=float, default=60.0, help="damping constant of the system")

parser.add_argument("--length", type=float, default=0.5, help="Length of Pole for single pendulum system m")
parser.add_argument("--gravity", type=float, default=9.8, help="Acceleration due to gravity m/s².")
args = parser.parse_args()


class SpringMassDamper:
    def __init__(self, m=args.mass, k=args.spring_constant, b=args.damping_constant):
        self.m = m
        self.k = k
        self.b = b

        A = np.array([[0, 1.], [-self.k / self.m, -self.b / self.m]])
        B = np.array([[0], [1 / self.m]])
        C = np.array([[1., 0]])
        D = np.array([[0]])
        self.sys = ct.ss(A, B, C, D)

    def step_response(self, plot=False, return_x=False):
        result = ct.step_response(self.sys)
        if plot:
            plt.plot(result.time, result.outputs)
            plt.show()
        return result

    def forced_response(self, num_traj=None, plot=False, input_traj=None, return_x=False):
        T = np.linspace(0, 100, len(input_traj))
        result = ct.forced_response(self.sys, T, input_traj, return_x=return_x)
        if plot:
            plt.plot(result.time, result.outputs)
            plt.show()
        return result


class SinglePendulum:
    def __init__(self, l=args.length, g=args.gravity):
        self.l = l
        self.g = g

        A = np.array([[0., 1], [-self.g/self.l, 0]])
        B = np.array([[0], [1]])
        C = np.array([[1, 0]])
        D = np.array([[0]])

        self.sys = ct.ss(A, B, C, D)

    def step_response(self):
        T = np.linspace(0, 100, 10000)
        t, y = ct.step_response(self.sys, T)
        plt.plot(t, y)
        plt.show()


