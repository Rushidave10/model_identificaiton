import matplotlib.pyplot as plt
import numpy as np

import models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Lf", type=float, default=2.3e-3, help="Inductance of filter H")
parser.add_argument("--Cf", type=float, default=10e-6, help="Capacitance of filter F")
parser.add_argument("--Rf", type=float, default=400e-3, help="Resistance inductor Ohm")
parser.add_argument("--w", type=float, default=314.16, help="Angular speed of rotating frame rad/s")
args = parser.parse_args()

ss = models.LC_filter(L=args.Lf,
                      C=args.Cf,
                      R=args.Rf,
                      Omega=args.w,
                      )

result = ss.step_response()
plt.subplot(211)
plt.plot(result.outputs[0])
plt.subplot(212)
plt.plot(result.outputs[1])
plt.subplot(221)
plt.plot(result.outputs[2])
plt.subplot(222)
plt.plot(result.outputs[3])

plt.show()
