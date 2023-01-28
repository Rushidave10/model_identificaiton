import numpy as np
import matplotlib.pyplot as plt
import LC
import argparse
import math
import control as ct

parser = argparse.ArgumentParser()
parser.add_argument("--Lf", default=2.3e-3, type=float, help="Filter inductance (H)")
parser.add_argument("--Rf", default=400e-3, type=float, help="Filter inductor's Internal resistance (Ohm)")
parser.add_argument("--Cf", default=10e-6, type=float, help="Filter Capacitance (F)")
parser.add_argument("--Omega", default=314.16, type=float, help="Angular speed set equal to grid frequency (rad/s)")
args = parser.parse_args()

sys = LC.LCfilter(L=args.Lf,
                  R=args.Rf,
                  C=args.Cf,
                  Omega=args.Omega)

t_32 = 2 / 3 * np.array([[1, -0.5, -0.5],
                         [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3)],
                         ])

t_23 = 2 / 3 * np.array([[1, 0],
                         [-0.5, 0.5 * np.sqrt(3)],
                         [-0.5, -0.5 * np.sqrt(3)]
                         ])


def q(U, angle):
    cos = math.cos(angle)
    sin = math.sin(angle)
    return cos * U[0] - sin * U[1], sin * U[0] + cos * U[1]


def q_inv(U, angle):
    cos = math.cos(angle)
    sin = math.sin(angle)
    return -sin * U[0] - cos * U[1], cos * U[0] - sin * U[1]


def abc_to_dq0(abc, angle):
    alphabeta = np.matmul(t_32, abc)
    return q_inv(alphabeta, angle)


def dq_to_alphabeta(dq, angle):
    return q(dq, angle)


t = np.arange(0, 0.5, 0.0001)
wt = 2 * np.pi * args.Omega * t
phase_B = 120.0 * np.pi / 180.0
phase_C = 240.0 * np.pi / 180.0
Vm = 230.0
ABC_in = Vm * np.array([np.cos(wt),
                        np.cos(wt + phase_B),
                        np.cos(wt + phase_C)])

dq = []
dqz_inv = []
for i in range(len(t)):
    dq.append(abc_to_dq0(ABC_in.T[i],  wt[i] - np.pi/2))

data = sys.forced_response(T=t, inputs=np.array(dq).T)
vd = data.outputs[2].T
vq = data.outputs[3].T

abc = []
for i in range(len(t)):
    abc.append(dq_to_alphabeta(data.outputs[2:].T[i], wt[i] - np.pi/2))

abc = np.matmul(t_23, np.array(abc).T)

plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot(t, ABC_in[0], label="A")
plt.plot(t, ABC_in[1], label="B")
plt.plot(t, ABC_in[2], label="C")
plt.title("Input Voltage - Three phase 120° apart")
plt.legend()

plt.subplot(222)
plt.plot(t, np.array(dq).T[0], label="d-axis")
plt.plot(t, np.array(dq).T[1], label="q-axis")
plt.title("Input Voltage - Rotating dq-frame")
plt.legend()

plt.subplot(223)
plt.plot(t, data.outputs[2].T, label="d-axis")
plt.plot(t, data.outputs[3].T, label="q_axis")
plt.title("Output Voltage - Rotating dq-frame")
plt.legend()

plt.subplot(224)
plt.plot(t, abc[0], label="A")
plt.plot(t, abc[1], label="B")
plt.plot(t, abc[2], label="C")
plt.title("Output Voltage - Three phase 120° apart")
plt.show()
