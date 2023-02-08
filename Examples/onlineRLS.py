import numpy as np
import matplotlib.pyplot as plt
import padasip as pa
import gpflow

def measure_x():
    x = np.random.random(1)
    return x


def measure_y(x):
    y_ = np.sin(x)
    return y_


t = np.arange(0, 0.5, 0.1e-3)
omega = 2
log_y = np.zeros(len(t))
log_y_ = np.zeros(len(t))
wt = 2 * np.pi * omega * t
X = wt
Y = np.sin(X)


def plot():
    plt.title("Adaptation")
    plt.xlabel("samples - k")
    plt.plot(log_y, "b", label="Target")
    plt.plot(log_y_, "g", label="Predicted")
    plt.legend()
    plt.show()


estimator = pa.filters.FilterRLS(1, mu=0.5)
for k in range(len(t)):
    # x = measure_x()
    # y_= estimator.predict(x)

    x = X[k]
    y_ = estimator.predict(X[k])

    # y = measure_y(x)
    y = Y[k]
    estimator.adapt(np.array([y]), np.array([x]))
    log_y[k] = y
    log_y_[k] = y_

# plot()
X1 = np.expand_dims(np.random.choice(wt, 3), axis=1)

Y1 = np.sin(X1)
model = gpflow.models.GPR((X1, Y1),
                          kernel=gpflow.kernels.SquaredExponential())

opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

f_mean, f_var = model.predict_f(wt[:, None], full_cov=False)
y_mean, y_var = model.predict_y(wt[:, None])

f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)


plt.plot(X1, Y1, "kx", mew=2, label="input data")
plt.plot(wt, f_mean, "-", color="C0", label="mean")
plt.plot(wt, f_lower, "--", color="C0", label="f 95% confidence")
plt.plot(wt, f_upper, "--", color="C0")
plt.fill_between(
    wt[:], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1
)
plt.legend()
plt.show()

