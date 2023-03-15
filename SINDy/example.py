import numpy as np
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ
from pysindy.optimizers import SR3

t = np.linspace(0, 1, 100)
x = 3 * np.exp(-2 * t)
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1)

feature_library = PolynomialLibrary(degree=2, interaction_only=False)

model = ps.SINDy(optimizer=STLSQ(),
                 feature_library=feature_library,
                 feature_names=["x", "y"],
                 discrete_time=False)
model.fit(X, t=t)

model.print()
model.coefficients()
