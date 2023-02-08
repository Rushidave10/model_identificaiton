import matplotlib.pyplot as plt
import numpy as np
import control as ctl
import control.optimal as opt


m = 0.1  # mass of pendulum (kg)
l = 0.5  # length of pendulum (m)
g = 9.8  # acceleration due to gravity (m/s^2)

# Define the state-space matrices
A = np.array([[0, 1], [g/l, 0]])  # state transition matrix
B = np.array([[0], [1/(m*l)]])    # input matrix
C = np.array([[1, 0]])            # output matrix
D = np.array([0])                 # feedforward matrix

# Define the state-space model
sys = ctl.ss(A, B, C, D, 0.2)
N = 20  # prediction horizon

# Define the weight matrices for the cost function
Q = np.array([[1, 0], [0, 0]])  # state weight matrix
R = np.array([1])               # input weight matrix

# Define the constraints on the states and inputs
xmin = np.array([-np.pi, -10])
xmax = np.array([np.pi, 10])
umin = np.array([-20])
umax = np.array([20])

# Define the initial state of the pendulum
x0 = np.array([np.pi, 0])

# Define the MPC controller
constraints = [opt.input_range_constraint(sys, [-20], [20])]
cost = opt.quadratic_cost(sys, Q, R, x0=x0)
ctrl = opt.create_mpc_iosystem(sys, np.arange(0, 6)*0.2, cost, constraints)

loop = ctl.feedback(sys, ctrl, 1)
