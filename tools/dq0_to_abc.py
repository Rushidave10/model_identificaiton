import matplotlib.pyplot as plt
import numpy as np

end_time = 1.0
num_pts = 500
t = np.linspace(0, end_time, num_pts)
wt = 2 * np.pi * 10 * t

amp = 230.0
rad_b = 120.0 * np.pi / 180
rad_c = 240.0 * np.pi / 180

A = amp * np.sin(wt)
B = amp * np.sin(wt + rad_b)
C = amp * np.sin(wt + rad_c)

abc = np.vstack((A, B, C)).T
t_32 = 2/3 * np.array([[1, -0.5, -0.5],
                       [0, 0.5*np.sqrt(3), -0.5*np.sqrt(3)],
                       ])
u = []
v = []
for i in range(500):
    u.append(np.matmul(t_32, abc[i]))
for j in range(500):
    t_22 = np.array([[ np.cos(wt[j]), -np.sin(wt[j])],
                     [np.sin(wt[j]), np.cos(wt[j])]])
    v.append(np.matmul(t_22, u[j]))



theta = 0

# T = (2 / 3) * np.array([[np.cos(theta), np.cos(theta - 2 * np.pi / 3), np.cos(theta - 4 * np.pi / 3)],
#                         [-np.sin(theta), -np.sin(theta - 2 * np.pi / 3), -np.sin(theta - 4 * np.pi / 3)],
#                         [1 / 2, 1 / 2, 1 / 2]
#                         ])
#
# for i in range(500):
#     d.append(np.matmul(T, abc[i]))




plt.subplot(211)
plt.plot(t, A)
plt.plot(t, B)
plt.plot(t, C)

plt.subplot(212)
# plt.plot(t, u)
plt.plot(t, v)
# plt.plot(t, w)
plt.show()
