import padasip as pa
import models
import numpy as np
import matplotlib.pyplot as plt

sys = models.SpringMassDamper()
data = sys.step_response()

inputs = np.expand_dims(data.inputs, axis=1)
states = np.hstack((np.transpose(data.states), inputs))

RLS_estimator1 = pa.filters.FilterRLS(n=3, mu=0.1, w='random')
RLS_estimator2 = pa.filters.FilterRLS(n=3, mu=0.1, w='random')

x1_dot = np.diff(data.states[0], prepend=0)
x2_dot = np.diff(data.states[1], prepend=0)

x1_dot_cap, e1, w1 = RLS_estimator1.run(x1_dot.T, states)
x2_dot_cap, e2, w2 = RLS_estimator2.run(x2_dot.T, states)

plt.figure(figsize=(15, 9))
plt.subplot(211)
plt.title("x1_dot")
plt.plot(x1_dot, "b", label="x1_dot")
plt.plot(x1_dot_cap, "g", label="estimate")
plt.legend()

plt.subplot(212)
plt.title("x2_dot")
plt.plot(x2_dot, "b", label="x2_dot")
plt.plot(x2_dot_cap, "g", label="estimate")
plt.legend()
plt.show()
