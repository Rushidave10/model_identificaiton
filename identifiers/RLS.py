import numpy as np
import matplotlib.pyplot as plt
import LC
import argparse
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--Lf", default=2.3e-3, type=float, help="Filter inductance (H)")
parser.add_argument("--Rf", default=400e-3, type=float, help="Filter inductor's Internal resistance (Ohm)")
parser.add_argument("--Cf", default=10e-6, type=float, help="Filter Capacitance (F)")
parser.add_argument("--Omega", default=314.16, type=float, help="Angular speed set equal to grid frequency (rad/s)")
parser.add_argument("--LoadCurrent", default=5.0, type=float, help="Current expected at load side (Amp)")
args = parser.parse_args()

sys = LC.LCfilter(L=args.Lf,
                  R=args.Rf,
                  C=args.Cf,
                  Omega=args.Omega)

t = np.arange(0, 0.5, 0.1e-3)
wt = 2 * np.pi * args.Omega * t
phase_B = 120.0 * np.pi / 180.0
phase_C = 240.0 * np.pi / 180.0
Vm = 230.0
Im = args.LoadCurrent
V_in = Vm * np.array([np.cos(wt),
                      np.cos(wt + phase_B),
                      np.cos(wt + phase_C)])

I_out = Im * np.array([np.cos(wt),
                       np.cos(wt + phase_B),
                       np.cos(wt + phase_C)])

dq_V, dq_I = [], []
dqz_inv = []
for i in range(len(t)):
    dq_V.append(abc_to_dq(V_in.T[i], wt[i] - np.pi / 2))
    dq_I.append(abc_to_dq(I_out.T[i], wt[i] - np.pi / 2))

data = sys.forced_response(T=t, inputs=np.array(np.hstack((dq_V, dq_I))).T)

# vd = data.outputs[2].T
# vq = data.outputs[3].T

abc_V = []
abc_I = []
for i in range(len(t)):
    abc_V.append(dq_to_abc(data.outputs[2:].T[i], wt[i] - np.pi / 2))
    abc_I.append(dq_to_abc(data.outputs[:2].T[i], wt[i] - np.pi/2))

abc_V = np.array(abc_V).T
abc_I = np.array(abc_I).T
plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot(t, V_in[0], label="A")
plt.plot(t, V_in[1], label="B")
plt.plot(t, V_in[2], label="C")
plt.title("Input Voltage - Three phase 120° apart")
plt.legend()

plt.subplot(222)
plt.plot(t, np.array(dq_V).T[0], label="d-axis")
plt.plot(t, np.array(dq_V).T[1], label="q-axis")
plt.title("Input Voltage - Rotating dq-frame")
plt.legend()

plt.subplot(223)
plt.plot(t, data.outputs[2].T, label="d-axis")
plt.plot(t, data.outputs[3].T, label="q_axis")
plt.title("Output Voltage - Rotating dq-frame")
plt.legend()

plt.subplot(224)
plt.plot(t, abc_V[0], label="A")
plt.plot(t, abc_V[1], label="B")
plt.plot(t, abc_V[2], label="C")
plt.title("Output Voltage - Three phase 120° apart")
plt.show()


plt.subplot(221)
plt.plot(t, I_out[0], label="Current a-axis")
plt.plot(t, I_out[1], label="Current b-axis")
plt.plot(t, I_out[2], label="Current c-axis")
plt.legend()


plt.subplot(222)
plt.plot(t, np.array(dq_I).T[0], label="Current d-axis")
plt.plot(t, np.array(dq_I).T[1], label="Current q-axis")
plt.legend()

plt.subplot(223)
plt.plot(t, abc_I[0], label="Current A")
plt.plot(t, abc_I[1], label="Current B")
plt.plot(t, abc_I[2], label="Current C")
plt.legend()

plt.subplot(224)
plt.plot(t, data.outputs[0].T, label="Current d-axis")
plt.plot(t, data.outputs[1].T, label="Current q-axis")
plt.legend()

plt.show()


