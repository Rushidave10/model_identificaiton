import control as ct
import matplotlib.pyplot as plt
import numpy as np
import control.optimal as opt
import padasip as pa

A = np.array([[1, 0.5],
              [0, 1]])

B = np.array([[0.125],
              [0.5]])

C = np.diag(np.ones(2))
x0 = np.array([3, 0])
xf = np.array([-3, 0])

sys = ct.ss(A, B, C, 0, 0.5)
model = ct.ss2io(sys)
in_constraints = opt.input_range_constraint(sys, -0.5, 0.5)
term_constraints = opt.state_range_constraint(sys, xf, xf)
state_constraints = opt.state_range_constraint(sys, [-5, -1], [5, 1])
all_constraints = [in_constraints, state_constraints]

Q = np.diag([1, 1])
R = np.array([0.5])
S = Q
traj_cost = opt.quadratic_cost(sys=sys, Q=Q, R=R, x0=x0, u0=np.array([0]))
terminal_cost = opt.quadratic_cost(sys, S, 0, x0=xf, u0=[0])
timepts = np.linspace(0, 10, 21, endpoint=True)
result = opt.solve_ocp(sys, timepts, X0=x0, cost=traj_cost,
                       constraints=all_constraints,
                       return_states=True,
                       terminal_constraints=term_constraints
                       )
t_eval = np.linspace(0, 5, 11, endpoint=True)
resp = ct.input_output_response(sys, timepts,
                                result.inputs, x0, t_eval=timepts)
t, y, u = resp.time, resp.outputs, resp.inputs

# Online RLS:

x1_estimator = pa.filters.FilterRLS(3, mu=0.5)
x2_estimator = pa.filters.FilterRLS(3, mu=0.5)
log_pred_x1 = np.zeros(len(t))
log_pred_x2 = np.zeros(len(t))
for k in range(len(t)):
    thruster = np.array(u[0][k])
    states = y[:, k]
    matx = np.array([states[0], states[1], thruster])
    x1_ = x1_estimator.predict(matx)
    x2_ = x2_estimator.predict(matx)
    if k != 20:
        x1 = y[0, k + 1]
        x2 = y[1, k + 1]
    else:
        x1 = y[0, k]
        x2 = y[0, k]
    x1_estimator.adapt(x1, matx)
    x2_estimator.adapt(x2, matx)
    log_pred_x1[k] = x1_
    log_pred_x2[k] = x2_

A_pred = [x1_estimator.w[:2], x2_estimator.w[:2]]

# Using GPR

plt.subplot(3, 1, 1)
plt.plot(t, y[0])
plt.plot(t, log_pred_x1, '--')
plt.ylabel("Theta[rad]")
plt.title("Angular Position")
plt.plot(x0[0], 'ro', t[-1], xf[0], 'ro')

plt.subplot(3, 1, 2)
plt.plot(t, y[1])
plt.plot(t, log_pred_x2, '--')
plt.ylabel("Theta_dot[rad/s]")
plt.title("Angular Velocity")

plt.subplot(313)
plt.plot(t, u.T)
plt.ylabel("F[N]")
plt.title("Thruster")
plt.tight_layout()
plt.show()

HEATMAP = False
if HEATMAP:
    plt.subplot(121)
    plt.imshow(A, cmap='hot', interpolation='nearest')
    plt.title("True system Matrix")
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(A_pred, cmap='hot', interpolation='nearest')
    plt.title("Estimated system Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
