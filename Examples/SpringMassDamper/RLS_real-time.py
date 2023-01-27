import padasip as pa
from identifiers import LC
import matplotlib.pyplot as plt
import numpy as np

sys = models.SpringMassDamper()
# data = sys.step_response(return_x=True)
data = sys.forced_response(plot=False,
                           input_traj=np.concatenate((2*np.ones(100), 3*np.ones(100))),
                           return_x=True)

inputs = np.hstack((data.states.T, np.expand_dims(data.inputs, axis=1)))
desired = np.diff(data.states, prepend=[[0], [0]]).T
plt.plot(data.states.T, label='states')
plt.plot(desired, label='state_dot')
plt.legend()
RLS_estimator = pa.filters.FilterRLS(n=3, mu=0.1)
RLS_estimator2 = pa.filters.FilterRLS(n=3, mu=0.1)
N = 199
log_d = np.zeros(N)
log_y = np.zeros(N)

for k in range(N):
    x = inputs[k]
    y = RLS_estimator.predict(x)

    d = desired[k][0]
    RLS_estimator.adapt(d, x)

    log_d[k] = d
    log_y[k] = y

plt.figure(figsize=(15, 6))
plt.subplot(211)
plt.title("Adaptation")
plt.xlabel("samples - k")
plt.plot(log_d, "b", label="d - target")
plt.plot(log_y, "g", label="y - output")
plt.legend()
plt.subplot(212)
plt.title("Filter error")
plt.xlabel("samples - k")
plt.plot(log_d-log_y, "r", label="e - error [dB]")
plt.legend()
plt.tight_layout()
plt.show()
