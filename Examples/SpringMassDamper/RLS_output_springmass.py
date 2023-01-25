import padasip as pa
import models
import matplotlib.pyplot as plt
import numpy as np

sys = models.SpringMassDamper()
data = sys.step_response(return_x=True)
outputs = np.expand_dims(data.outputs, axis=1)
states = np.transpose(data.states)

RLS_estimator = pa.filters.FilterRLS(n=2, mu=0.1, w='random')
y, e, w = RLS_estimator.run(outputs, states)


plt.figure(figsize=(15, 9))
plt.subplot(211)
plt.title("Adaptation")
plt.xlabel("samples - k")
plt.plot(data.outputs, "b", label="System_outputs")
plt.plot(y, "g", label="predicted outputs")
plt.legend()
plt.subplot(212)
plt.title("Filter error")
plt.xlabel("samples - k")
plt.plot(e, "r", label="e - error [dB]")
plt.legend()
plt.tight_layout()
plt.show()

