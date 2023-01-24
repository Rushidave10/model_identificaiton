import control as ct
import matplotlib.pyplot as plt
import numpy as np
import control.optimal as obc


def vehicle_update(t, x, u, params):
    l = params.get('wheelbase', 3.)
    phimax = params.get('maxsteer', 0.5)

    phi = np.clip(u[1], -phimax, phimax)

    return np.array([np.cos(x[2]) * u[0],
                     np.sin(x[2]) * u[0],
                     (u[0] / l) * np.tan(phi)
                     ])


def vehicle_output(t, x, u, params):
    return x


vehicle = ct.NonlinearIOSystem(vehicle_update,
                               vehicle_output,
                               states=3,
                               name='vehicle',
                               inputs=('v', 'phi'),
                               outputs=('x', 'y', 'theta')
                               )

x0 = np.array([0., -2., 0.])
u0 = np.array([10., 0.])

xf = np.array([100., 2., 0.])
uf = np.array([10., 0.])

Tf = 10

Q = np.diag([0, 0, 0.1])
R = np.diag([1, 1])
P = np.diag([1000, 1000, 1000])

traj_cost = obc.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)
term_cost = obc.quadratic_cost(vehicle, P, 0, x0=xf)

constraints = [obc.input_range_constraint(vehicle, [8, -0.1], [12, 0.1])]

timepts = np.linspace(0, Tf, 10, endpoint=True)
result = obc.solve_ocp(vehicle,
                       timepts,
                       x0,
                       traj_cost,
                       constraints,
                       terminal_cost=term_cost,
                       initial_guess=u0
                       )

resp = ct.input_output_response(
    vehicle, timepts, result.inputs, x0,
    t_eval=np.linspace(0, Tf, 100))
t, y, u = resp.time, resp.outputs, resp.inputs

plt.subplot(3, 1, 1)
plt.plot(y[0], y[1])
plt.plot(x0[0], x0[1], 'ro', xf[0], xf[1], 'ro')
plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.subplot(3, 1, 2)
plt.plot(t, u[0])
plt.axis([0, 10, 9.9, 10.1])
plt.xlabel("t [sec]")
plt.ylabel("u1 [m/s]")

plt.subplot(3, 1, 3)
plt.plot(t, u[1])
plt.axis([0, 10, -0.015, 0.015])
plt.xlabel("t [sec]")
plt.ylabel("u2 [rad/s]")

plt.suptitle("Lane change manuever")
plt.tight_layout()
plt.show()
