import control as ct
import matplotlib.pyplot as plt
import numpy as np

sys = ct.rss(4, 1, 1)
t, y = ct.forced_response(sys, T=np.linspace(0, 1, 1000), U=np.ones(1000))
t2, y2 = ct.step_response(sys)
plt.plot(t2, y2)
plt.legend("Step Response")
plt.plot(t, y)
ct.bode_plot(sys)
plt.show(block=True)
