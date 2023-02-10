import control as ct
import matplotlib.pyplot as plt
import numpy as np
import control.optimal as opt
import time

A = np.array([[1, 0.5],
              [0, 1]])

B = np.array([[0.125],
              [0.5]])

C = np.diag(np.ones(2))
x0 = np.array([[3],
               [0]])

sys = ct.ss(A, B, C, 0, 0.5)
model = ct.ss2io(sys)
in_constraints = opt.input_range_constraint(sys, -0.5, 0.5)
state_constraints = opt.state_range_constraint(sys, [0, -1], [0, 1])
all_constraints = [in_constraints, state_constraints]

Q = np.diag(np.ones(2))
R = np.array([1])
cost = opt.quadratic_cost(sys=sys, Q=Q, R=R, x0=x0, u0=np.array([0]))
timepts = np.linspace(0, 10, 21, endpoint=True)
ctrl = opt.create_mpc_iosystem(model, timepts, cost, all_constraints)

loop = ct.feedback(sys, ctrl, 1)

Nsim = 60
tout, xout = ct.input_output_response(loop, timepts, X0=x0)

for i, y in enumerate(C@xout):
    plt.plot(tout, y)
plt.title('outputs')
plt.show()

